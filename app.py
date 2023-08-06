import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import csv
import os
from datetime import datetime
import base64
import pandas as pd
from vg16 import vgmain
# Function to predict the species and accuracy
def predict_species(img):
    model = ResNet50(weights='imagenet')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    decoded_preds = decode_predictions(preds, top=1)[0]
    species_names = [pred[1].replace('_', ' ') for pred in decoded_preds]
    accuracies = [pred[2] for pred in decoded_preds]

    return species_names[0], accuracies[0]

# Function to save the results in a CSV file with the current date and time
def save_results_to_csv(results):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")  # Remove colons from the timestamp
    file_name = "results.csv"
    with open(file_name, "a+", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["Date", "Time", "Species Name", "Accuracy"])
        for result in results:
            writer.writerow([result[0], result[1], result[2], result[3]])
    return file_name
# Streamlit app
def display_previous_results():
    try:
        df = pd.read_csv("results.csv")
        st.write("Previous Results:")
        st.write(df)
    except FileNotFoundError:
        st.warning("No previous results found.")


if "results" in st.session_state:
    with open(st.session_state.results, "rb") as f:
        data = f.read()
    st.sidebar.download_button("Download CSV", data, file_name=os.path.basename(st.session_state.results), key="download_button")

# Streamlit app
def main():
    st.title("Animal Species Prediction")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            species_name, accuracy = predict_species(img)
            st.write(f"Species: {species_name}")
            st.write(f"Accuracy: {accuracy}")

            # Save the results with the current date and time
            results = [(datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"), species_name, accuracy)]
            file_name = save_results_to_csv(results)
            st.success(f"Results saved to {file_name}")

            # Store the CSV file path in session state
            st.session_state.results = file_name
    # Add buttons in the sidebar
#st.sidebar.button("Download CSV", on_click=download_csv)
st.sidebar.button("Display Previous Results", on_click=display_previous_results)
selected_model = st.sidebar.selectbox("Select Model", ["VGG16", "ResNet50"])

# Function to download the CSV file


if __name__ == "__main__":
    main()
