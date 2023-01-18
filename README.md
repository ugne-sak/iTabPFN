# iTabPFN


Interfeature TabPFN implementation. 

Please ensure that Python==3.7.15 is installed in your environment.

To install the requirements file with 

```bash
pip install -r requirements.txt
conda install -c conda-forge lightgbm
```

## Demo
For demonstration of iTabPFN performance on real-world classification task, please have a look at the demo.ipynb in the /tabpfn folder.
The file prior_diff_real_checkpoint_Euler_T2_1L_bH_n_2_n_2_epoch_32.cpkt holds the parameters of iTabPFN, learned during pre-training. Ensure that the base path leads to the /tabpfn folder.

The TabPFNClassifier class has a similar interface to other baselines in sklearn.

## Training
To begin training a new model please navigate to the tabpfn/PriorFittingCustomPrior.ipynb notebook, where  parameters to train a customised model can be set. The whole file until the last two cells can be run, where either function get_model or train_function starts training, periodically saving model parameters to the tabpfn/model_diffs folder. This file can be loaded using the TabPFNClassifier class as show in the demo.ipynb file.

## Evaluation metrics: plots and tables
To reproduce the performance evaluation metrics of iTabPFN and other baseline methods please run the Evaluation.ipynb notebook. This file contains all plots and tables that was presented in the report. Note that it is expected to get warnings of the type "Execution failed <dataset name>" or "Warning not all datasets generated for <results file name>" as some methods did not produce meaningful results for lower time budgets and their results files are not loaded.

## Generating baseline evaluation results files
The Results.ipynb notebook contains the script for generating performance metrics for baseline methods on the 30 small classification tasks from OpenML-CC18 suite mentioned in the report. 




