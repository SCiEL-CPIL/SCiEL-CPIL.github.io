import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    # Ensure 2D array
    X = np.asarray(X_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=float)
    
    # Handle potential NaNs/Infs
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n_samples, n_features = X.shape
    
    # Standardize X and y
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0 # Avoid division by zero
    
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Compute pairwise distances for heuristic gamma
    # Using cdist is efficient
    D = cdist(X_norm, X_norm, metric='euclidean')
    # Get upper triangle to avoid 0s
    triu_indices = np.triu_indices(n_samples, k=1)
    dists = D[triu_indices]
    if len(dists) > 0:
        median_dist = np.median(dists)
    else:
        median_dist = 1.0
    
    # Heuristic gamma
    gamma_heuristic = 1.0 / (n_features * median_dist**2)
    
    # Define grid for hyperparameters
    # Gamma around heuristic
    gamma_grid = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    # Scale gamma grid relative to heuristic if heuristic is very different?
    # Better to just use a log-spaced grid covering a wide range.
    gamma_grid = np.logspace(-2, 3, 10) # 0.01 to 1000
    
    alpha_grid = np.logspace(-1, 4, 8) # 0.1 to 10000
    
    best_mse = np.inf
    best_gamma = gamma_heuristic
    best_alpha = 1.0
    
    # Simple 5-Fold Cross Validation
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42) # For reproducibility in CV split
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    # Pre-compute nothing, just loop. N=200 is small.
    for gamma in gamma_grid:
        # Compute kernel matrix for training set once? No, we need subsets.
        # But we can compute the full K matrix once for this gamma?
        # K = exp(-gamma * D^2)
        K_full = np.exp(-gamma * D**2)
        
        for alpha in alpha_grid:
            cv_mse_sum = 0.0
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                K_val_val = K_full[np.ix_(val_idx, val_idx)]
                K_val_train = K_full[np.ix_(val_idx, train_idx)]
                K_train_train = K_full[np.ix_(train_idx, train_idx)]
                y_val = y_norm[val_idx]
                y_train = y_norm[train_idx]
                
                # Solve (K_train + alpha I) beta = y_train
                # K_train is symmetric positive definite (plus alpha I)
                try:
                    beta = np.linalg.solve(K_train_train + alpha * np.eye(len(train_idx)), y_train)
                    y_pred_val = K_val_train @ beta
                    mse = np.mean((y_val - y_pred_val)**2)
                    cv_mse_sum += mse
                except np.linalg.LinAlgError:
                    cv_mse_sum += np.inf
                    break
            
            mean_cv_mse = cv_mse_sum / n_folds
            if mean_cv_mse < best_mse:
                best_mse = mean_cv_mse
                best_gamma = gamma
                best_alpha = alpha
                
    # Retrain on full dataset with best parameters
    K_full_best = np.exp(-best_gamma * D**2)
    try:
        beta = np.linalg.solve(K_full_best + best_alpha * np.eye(n_samples), y_norm)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular (unlikely with alpha > 0)
        beta, *_ = np.linalg.lstsq(K_full_best + best_alpha * np.eye(n_samples), y_norm, rcond=None)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_norm = (X_test - X_mean) / X_std
        
        # Compute kernel between test and train
        D_test = cdist(X_test_norm, X_norm, metric='euclidean')
        K_test_train = np.exp(-best_gamma * D_test**2)
        
        y_pred_norm = K_test_train @ beta
        y_pred = y_pred_norm * y_std + y_mean
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return y_pred

    return predict
