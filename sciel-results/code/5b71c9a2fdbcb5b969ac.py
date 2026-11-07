import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    # Start time for this equation
    start_time = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Flatten if necessary, though task says 1 input
    if X_train.ndim == 2:
        X_train = X_train.flatten()
    else:
        X_train = X_train.reshape(-1)
        
    y_train = y_train.reshape(-1)
    
    n_samples = len(X_train)
    
    # Normalize inputs to [0, 1] to make sigma selection easier
    x_min = X_train.min()
    x_max = X_train.max()
    if x_max == x_min:
        x_max = x_min + 1e-6
    X_train_norm = (X_train - x_min) / (x_max - x_min)
    
    # Grid search for RBF Kernel Ridge Regression hyperparameters
    # sigma: length scale, lambda: regularization
    sigmas = np.logspace(-1, 1, 15)  # 0.1 to 10
    lambdas = np.logspace(-2, 2, 9)  # 0.01 to 100
    
    best_cv_error = np.inf
    best_sigma = 1.0
    best_lambda = 1.0
    
    # K-Fold Cross Validation (5-fold)
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42) # Fixed seed for determinism in CV split
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    # Pre-compute kernel matrices for all training pairs? 
    # K matrix is 200x200, which is small. We can compute it once per sigma.
    
    for sigma in sigmas:
        # Compute full kernel matrix for this sigma
        # X_train_norm is (200,)
        # Diff is (200, 200)
        diff = X_train_norm[:, None] - X_train_norm[None, :]
        K = np.exp(-0.5 * (diff / sigma) ** 2)
        
        for lam in lambdas:
            cv_errors = []
            for i in range(n_folds):
                val_idx = fold_indices[i]
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                
                K_train = K[np.ix_(train_idx, train_idx)]
                K_val = K[np.ix_(val_idx, train_idx)]
                
                y_tr = y_train[train_idx]
                y_va = y_train[val_idx]
                
                # Solve (K + lambda*I) alpha = y
                A = K_train + lam * np.eye(len(train_idx))
                try:
                    alpha = solve(A, y_tr)
                    y_pred = K_val @ alpha
                    mse = np.mean((y_pred - y_va) ** 2)
                    cv_errors.append(mse)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_error = np.mean(cv_errors)
            
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_sigma = sigma
                best_lambda = lam
                
        # Time check
        if time.time() - start_time > 60:
            break
            
    # Final fit with best parameters
    diff = X_train_norm[:, None] - X_train_norm[None, :]
    K_final = np.exp(-0.5 * (diff / best_sigma) ** 2)
    A_final = K_final + best_lambda * np.eye(n_samples)
    
    try:
        alpha_final = solve(A_final, y_train)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (should be rare with ridge)
        alpha_final = np.linalg.lstsq(A_final, y_train, rcond=None)[0]
        
    # Store necessary data for prediction
    x_min_val = x_min
    x_max_val = x_max
    sigma_val = best_sigma
    alpha_val = alpha_final
    X_train_norm_val = X_train_norm
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X.flatten()
        else:
            X = X.reshape(-1)
            
        # Normalize input
        X_norm = (X - x_min_val) / (x_max_val - x_min_val)
        
        # Compute kernel between test points and training points
        diff = X_norm[:, None] - X_train_norm_val[None, :]
        K_test = np.exp(-0.5 * (diff / sigma_val) ** 2)
        
        y_pred = K_test @ alpha_val
        return y_pred

    return predict
