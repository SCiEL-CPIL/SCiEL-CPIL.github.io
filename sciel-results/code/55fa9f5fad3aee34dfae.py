import numpy as np
import time
from scipy.special import comb

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 100.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    if len(x) == 0:
        return np.zeros(6)
        
    # Standardize x
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x < 1e-10:
        std_x = 1.0
    x_std = (x - mean_x) / std_x
    
    # Build Vandermonde matrix for degree 5 (increasing powers)
    V = np.vander(x_std, N=6, increasing=True)
    
    # Precompute Gram matrix and vector
    G = V.T @ V
    g = V.T @ y
    
    # Regularization parameter search
    lambdas = np.logspace(-6, 6, 50)
    
    # Cross-validation setup
    n_samples = len(x)
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    folds[-1] = indices[(n_folds-1)*fold_size:] # Handle remainder
    
    best_lambda = 1.0
    best_cv_error = np.inf
    best_coeffs_std = np.zeros(6)
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        cv_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            # We need G_train and g_train for the subset
            # This is expensive if done naively. 
            # Alternative: Use leave-one-out or just simple holdout if n is large?
            # Actually, recomputing G and g for each fold is O(N^2 * 6^2). 
            # If N is large (e.g. 100k), this might be slow.
            # However, V is N x 6. G is 6x6.
            # V_train is N_train x 6. G_train = V_train.T @ V_train.
            # This is fast.
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            G_train = V_train.T @ V_train
            g_train = V_train.T @ y_train
            
            A = G_train + lam * np.eye(6)
            try:
                w = np.linalg.solve(A, g_train)
                pred_val = V_val @ w
                error = np.mean((y_val - pred_val) ** 2)
                cv_errors.append(error)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)
                
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_lambda = lam
            # Store coeffs for this lambda (refit on all data)
            A_all = G + best_lambda * np.eye(6)
            try:
                best_coeffs_std = np.linalg.solve(A_all, g)
            except:
                pass

    # Refit with best lambda on all data
    A_all = G + best_lambda * np.eye(6)
    try:
        coeffs_std = np.linalg.solve(A_all, g)
    except:
        coeffs_std = best_coeffs_std

    # Transform coefficients from x_std to x
    alpha = 1.0 / std_x
    beta = -mean_x / std_x
    
    coeffs_orig = np.zeros(6)
    for j in range(6):
        sum_val = 0.0
        for i in range(j, 6):
            binom_val = comb(i, j, exact=True)
            term = coeffs_std[i] * binom_val * (alpha ** j) * (beta ** (i - j))
            sum_val += term
        coeffs_orig[j] = sum_val
        
    return coeffs_orig
