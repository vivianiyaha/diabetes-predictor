import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the diabetes dataset from Kaggle
diabetes_df = pd.read_csv('diabetes.csv')

# Sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Predictor', 'About', 'Profile'])

# Main content
if page == 'Predictor':
    st.title('Diabetes Predictor')
    
    # Split dataset into features and target variable
    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a random forest classifier
    clf = RandomForestClassifier()
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Function to predict diabetes
    def predict_diabetes(data):
        prediction = clf.predict(data)
        return prediction
    
    # User input for predictor
    st.header('Enter Your Health Information:')
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=79)
    bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.5)
    age = st.number_input('Age', min_value=21, max_value=81, value=30)
    
    # Create a data frame for prediction
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })
    
    # Predict button
    if st.button('Predict'):
        prediction = predict_diabetes(input_data)
        if prediction[0] == 0:
            st.success('You are not likely to have diabetes.')
        else:
            st.error('You are likely to have diabetes.')
            
elif page == 'About':
    st.title('About')
    st.write('The Diabetes Predictor is a web application built using Streamlit and a diabetes dataset from Kaggle. It employs a Random Forest Classifier model to predict the likelihood of diabetes based on key health metrics. The dataset used for training the model is sourced from Kaggle, ensuring transparency and reliability.')
    st.write('Users can input their health data into the prediction model interactive interface to receive instant predictions, empowering proactive healthcare management and promoting early intervention strategies. The Diabetes Predictor is committed to continuous improvement and welcomes feedback from users and the healthcare community.')    

elif page == 'Profile':
    st.title('Profile')
    st.write('Vivian Iyaha is a University of Port Harcourt graduate with a degree in Management. She is skilled in management and has a strong interest in emerging technologies like Machine Learning and Artificial Intelligence.')
    st.write('Vivian is seeking opportunities to apply her managerial expertise and learn more about these technologies. She is enthusiastic about collaborating on projects that leverage Machine Learning and Artificial Intelligence to solve real-world problems and drive innovation.')
    st.write('Connect with Vivian on [LinkedIn](https://www.linkedin.com/in/vivian-i-556499126/)')


