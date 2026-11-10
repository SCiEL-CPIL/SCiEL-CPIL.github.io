import numpy as np
from scipy.linalg import solve
import time

def fit(X_train, y_train):
    """
    Fits a Kernel Ridge Regression model with an RBF kernel.
    Uses a small hyperparameter search to optimize length scale and regularization.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s buffer for safety
    
    # 1. Preprocessing: Standardize inputs
    # This helps the RBF kernel work consistently across different scales
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    # Avoid division by zero
    X_std[X_std == 0] = 1.0
    
    X_train_std = (X_train - X_mean) / X_std
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_std = y_std if y_std != 0 else 1.0
    y_train_std = (y_train - y_mean) / y_std
    
    N, D = X_train_std.shape
    
    # 2. Hyperparameter Search
    # We use a train/val split to select hyperparameters
    # Since N=200 is small, we can do a simple grid search
    # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    # gamma = 1 / (2 * sigma^2)
    
    # Candidate gammas (inverse squared length scale)
    # Small gamma -> large sigma -> smooth
    # Large gamma -> small sigma -> wiggly
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    # Candidate alphas (regularization strength)
    alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    
    # Split for validation
    # Use a fixed split for determinism, or random with seed
    indices = np.arange(N)
    # Simple 80/20 split
    val_size = int(0.2 * N)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]
    
    X_tr = X_train_std[train_idx]
    y_tr = y_train_std[train_idx]
    X_val = X_train_std[val_idx]
    y_val = y_train_std[val_idx]
    
    best_val_mse = np.inf
    best_gamma = 1.0
    best_alpha = 1e-2
    
    # Pre-compute training kernel matrix for each gamma? No, gamma changes.
    # We can speed up by computing pairwise distances once, then applying gamma.
    # Distances squared: ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi.xj
    X_tr_sq = np.sum(X_tr**2, axis=1)
    X_val_sq = np.sum(X_val**2, axis=1)
    
    # For each gamma, compute K_tr and K_val
    # K_tr = exp(-gamma * D_tr)
    # D_tr = X_tr_sq[:, None] + X_tr_sq[None, :] - 2 * X_tr @ X_tr.T
    
    # To save time, we can limit the number of combinations if time is tight
    # But 9 * 6 = 54 combinations. Each involves matrix ops on 160x160 and 40x160.
    # This is very fast in NumPy.
    
    for gamma in gammas:
        # Compute train kernel
        D_tr = X_tr_sq[:, None] + X_tr_sq[None, :] - 2 * (X_tr @ X_tr.T)
        # Numerical stability: clip negative values
        D_tr = np.maximum(D_tr, 0)
        K_tr = np.exp(-gamma * D_tr)
        
        # Solve (K_tr + alpha I) w = y_tr
        # We can do this for each alpha
        for alpha in alphas:
            A = K_tr + alpha * np.eye(len(X_tr))
            try:
                w = solve(A, y_tr, assume_a='pos')
            except:
                # Fallback to lstsq if solve fails (shouldn't happen with alpha > 0)
                w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
            
            # Compute validation kernel
            D_val = X_val_sq[:, None] + X_tr_sq[None, :] - 2 * (X_val @ X_tr.T)
            D_val = np.maximum(D_val, 0)
            K_val = np.exp(-gamma * D_val)
            
            y_val_pred = K_val @ w
            val_mse = np.mean((y_val - y_val_pred)**2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_gamma = gamma
                best_alpha = alpha
        
        # Time check
        if time.time() > deadline - 10:
            break
    
    # 3. Refit on full training set with best hyperparameters
    gamma = best_gamma
    alpha = best_alpha
    
    X_all_sq = np.sum(X_train_std**2, axis=1)
    D_all = X_all_sq[:, None] + X_all_sq[None, :] - 2 * (X_train_std @ X_train_std.T)
    D_all = np.maximum(D_all, 0)
    K_all = np.exp(-gamma * D_all)
    
    A_all = K_all + alpha * np.eye(N)
    try:
        w_final = solve(A_all, y_train_std, assume_a='pos')
    except:
        w_final, *_ = np.linalg.lstsq(A_all, y_train_std, rcond=None)
    
    # 4. Define predict function
    def predict(X):
        # Standardize X
        X_std = (X - X_mean) / X_std
        X_test_sq = np.sum(X_std**2, axis=1)
        
        # Compute kernel between test and train
        D_test = X_test_sq[:, None] + X_all_sq[None, :] - 2 * (X_std @ X_train_std.T)
        D_test = np.maximum(D_test, 0)
        K_test = np.exp(-gamma * D_test)
        
        y_pred_std = K_test @ w_final
        
        # Inverse standardize
        y_pred = y_pred_std * y_std + y_mean
        
        return y_pred
    
    return predict
