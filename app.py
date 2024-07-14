import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('generator.keras')

def generate_face():
    noise = tf.random.normal((1, 100))
    face = model(noise, training=False)
    face = (face + 1) * 127.5
    face = tf.clip_by_value(face, 0, 255)
    face = tf.cast(face, tf.uint8)
    face = tf.squeeze(face, axis=0).numpy()
    return Image.fromarray(face)

def main():
    st.title('Human face generation')
    st.write('This app generates human faces using a GAN model.')
    st.write('Click the button below to generate a new face.')
    
    if st.button('Generate face'):
        # Generate and upscale the face
        face_image = generate_face()
        upscale_factor = 2
        new_size = (face_image.width * upscale_factor, face_image.height * upscale_factor)
        upscaled_face_image = face_image.resize(new_size, Image.BICUBIC)
        
        # Display the upscaled face
        st.image(upscaled_face_image)
    
    # Display the pre-existing GIF
    st.write('Transition:')
    gif_path = 'dcgan.gif'  # Replace with the path to your GIF file
    st.image(gif_path)

if __name__ == '__main__':
    main()
