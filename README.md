# Tax Revenue Prediction using Genetic Algorithms and Deep Neural Networks

This project predicts tax revenue from macroeconomic indicators using a hybrid Genetic Algorithm and Deep Neural Network approach. The Genetic Algorithm searches for a suitable neural network architecture, and the selected architecture is then retrained and used for final tax revenue prediction.

The project was developed as part of an academic machine learning assignment and includes source code, configuration files, dataset sample, trained model artifacts, result figures, and an IEEE-style project report.

## Project Overview

Tax revenue forecasting is important for public finance planning because it helps estimate future government income. Traditional statistical models may not fully capture nonlinear relationships between economic indicators and tax revenue. This project uses a Deep Neural Network to learn those nonlinear relationships and a Genetic Algorithm to reduce manual architecture tuning.

The model uses the following input features:

- GDP
- Inflation
- Population
- Imports
- Exports
- Corporate tax rate

The target variable is:

- Tax revenue

## Key Features

- Complete end-to-end machine learning pipeline
- Data loading, cleaning, splitting, and scaling
- Genetic Algorithm-based neural architecture search
- PyTorch-based Deep Neural Network regression model
- Model retraining and test-set evaluation
- Saved final model and scalers for inference
- Command-line prediction interface
- IEEE conference-format report in LaTeX
- GA fitness curve visualization

## Project Structure

```text
tax-revenue-ga-nn/
|-- configs/
|   `-- config.yaml
|-- data/
|   `-- raw/
|       `-- tax_sample.csv
|-- models/
|   |-- best_architecture.json
|   |-- final_model.pth
|   |-- final_X_scaler.pkl
|   |-- final_y_scaler.pkl
|   `-- model_test_eval.pth
|-- notebooks/
|   |-- 01_exploration.ipynb
|   |-- 02_preprocessing.ipynb
|   `-- 03_model_experiments.ipynb
|-- reports/
|   |-- tax_revenue_ieee_paper.tex
|   |-- tax_revenue_demo.pptx
|   `-- figures/
|       `-- ga_fitness_curve.png
|-- scripts/
|   |-- predict.py
|   |-- retrain_best_model.py
|   `-- run_ga_search.py
|-- src/
|   |-- data/
|   |   `-- load_data.py
|   |-- ga/
|   |   |-- chromosome.py
|   |   |-- operators.py
|   |   `-- population.py
|   |-- models/
|   |   `-- nn_models.py
|   |-- training/
|   |   |-- metrics.py
|   |   `-- train_eval.py
|   |-- utils/
|   |   `-- config.py
|   `-- visualization/
|       `-- ga_plots.py
|-- tests/
|   `-- test_basic.py
|-- requirements.txt
|-- run_all.py
`-- README.md
```

## Dataset

The sample dataset is located at:

```text
data/raw/tax_sample.csv
```

It contains 129 records and the following columns:

| Column | Description |
|---|---|
| `gdp` | Gross domestic product value |
| `inflation` | Inflation rate |
| `population` | Population indicator |
| `imports` | Import value |
| `exports` | Export value |
| `corporate_tax_rate` | Corporate tax rate |
| `tax_revenue` | Target tax revenue value |

The dataset is synthetic but structured to represent a macroeconomic tax revenue forecasting problem.

## Methodology

The program follows this workflow:

1. Load configuration values from `configs/config.yaml`.
2. Load the dataset from `data/raw/tax_sample.csv`.
3. Remove unsupported non-numeric columns or convert date-like columns to year values.
4. Separate input features and target tax revenue.
5. Split the dataset into training, validation, and testing sets.
6. Standardize input features and target values using `StandardScaler`.
7. Run Genetic Algorithm search to find a good neural network architecture.
8. Save the best architecture to `models/best_architecture.json`.
9. Retrain the selected architecture on training and validation data.
10. Evaluate the model on the held-out test set.
11. Train a final deployment model on all available data.
12. Save the final model and scalers.
13. Use the prediction script for new tax revenue predictions.

## Genetic Algorithm Search

The Genetic Algorithm searches over neural network architectures. Each chromosome represents:

- Hidden layer configuration
- Activation function

The search space used in the project includes:

```text
Hidden layers:
[64, 64]
[128, 64]
[128, 128]
[256, 128]
[256, 128, 64]

Activation functions:
ReLU
Tanh
```

Main GA settings:

| Parameter | Value |
|---|---:|
| Population size | 20 |
| Generations | 10 |
| Crossover probability | 0.9 |
| Mutation probability | 0.4 |
| Training epochs during search | 200 |

The validation Mean Squared Error is used as the fitness value. Lower validation loss means a better architecture.

## Best Model and Results

The best architecture found by the Genetic Algorithm was:

```json
{
    "layers": [64, 64],
    "activation": "tanh"
}
```

Final test-set performance:

| Metric | Value |
|---|---:|
| MSE | 4,193,052,107.50 |
| RMSE | 64,753.78 |
| MAE | 52,092.07 |
| R-squared | 0.8499 |

The R-squared value shows that the model explains approximately 85 percent of the variation in tax revenue on the test data.

## Installation

### 1. Clone or Download the Project

If using GitHub:

```bash
git clone <your-private-github-repo-link>
cd tax-revenue-ga-nn
```

If using a zip file, extract the zip and open the extracted project folder.

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## How to Run

Run all major steps using:

```bash
python run_all.py
```

This will:

1. Run the GA architecture search.
2. Save the best architecture.
3. Retrain the best model.
4. Save the final trained model and scalers.
5. Ask whether you want to run prediction.

## Run Individual Scripts

### Run Genetic Algorithm Search

```bash
python -m scripts.run_ga_search
```

Output:

```text
models/best_architecture.json
```

### Retrain Best Model and Evaluate

```bash
python -m scripts.retrain_best_model
```

Outputs:

```text
models/model_test_eval.pth
models/final_model.pth
models/final_X_scaler.pkl
models/final_y_scaler.pkl
```

### Run Prediction

```bash
python -m scripts.predict
```

The script asks for:

```text
gdp
inflation
population
imports
exports
corporate_tax_rate
```

It then prints the predicted tax revenue.

## Configuration

The main configuration file is:

```text
configs/config.yaml
```

Current configuration:

```yaml
population_size: 20
generations: 10
crossover_prob: 0.9
mutation_prob: 0.4

epochs: 200
batch_size: 16
learning_rate: 0.001

csv_path: "data/raw/tax_sample.csv"
target_column: "tax_revenue"

predict_features:
  - gdp
  - inflation
  - population
  - imports
  - exports
  - corporate_tax_rate
```

To use another dataset, update `csv_path`, `target_column`, and `predict_features`.

## Report and Figures

The IEEE-format LaTeX report is available at:

```text
reports/tax_revenue_ieee_paper.tex
```

The GA fitness curve is available at:

```text
reports/figures/ga_fitness_curve.png
```

To compile the report, open the `.tex` file in Overleaf or another LaTeX editor. Make sure the figure file remains inside `reports/figures/`.

## Requirements

The project uses:

```text
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
deap
tqdm
pyyaml
```

These packages are listed in `requirements.txt`.

## GitHub Upload Checklist

Before uploading to GitHub, make sure the repository includes:

- `README.md`
- `requirements.txt`
- `configs/config.yaml`
- `data/raw/tax_sample.csv`
- `src/`
- `scripts/`
- `reports/`
- `models/best_architecture.json`
- final model files, if allowed by the submission rules

Do not upload unnecessary files such as:

- `__pycache__/`
- `.venv/`
- `.ipynb_checkpoints/`
- temporary zip files
- local editor settings

## Creating a Zip File for Submission

From the parent folder, zip the complete project folder:

```text
tax-revenue-ga-nn/
```

The zip should preserve the folder structure. After extraction, the evaluator should be able to run:

```bash
pip install -r requirements.txt
python run_all.py
```

## Authors

Afrah Zainab  
Department of CSE - AIML  
PES University  

Abhi Arasi  
Department of CSE - AIML  
PES University  

Prof. Preethi S. J.  
Department of CSE - AIML  
PES University  

## Acknowledgment

The authors thank PES University and the Department of CSE - AIML for academic support and project guidance.
