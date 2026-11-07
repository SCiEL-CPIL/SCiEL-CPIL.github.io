import numpy as np
import time

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 25.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    X = X_train
    y = y_train
    n, d = X.shape
    
    # Standardize X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_scaled = (X - mu) / sigma
    
    # Ensure 1D handling if needed, though task says 1D
    x = X_scaled[:, 0]
    
    # Split for validation (80/20)
    # Use a fixed seed for reproducibility in selection
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = 40
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_tr = x[train_idx]
    y_tr = y[train_idx]
    X_val = x[val_idx]
    y_val = y[val_idx]
    
    best_val_error = np.inf
    best_params = None
    
    # Grid search for RBF kernel parameters
    # Gamma controls width, Lambda controls regularization
    # Log-spaced grids
    gammas = np.logspace(-2, 3, 10)  # 0.01 to 1000
    lambdas = np.logspace(-8, 2, 11) # 1e-8 to 100
    
    def rbf_kernel(X1, X2, gamma):
        # Efficient squared distance calculation
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
        X1_sq = np.sum(X1**2, axis=1)[:, None]
        X2_sq = np.sum(X2**2, axis=1)[None, :]
        sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)
    
    def solve_ridge_svd(K, y, lam):
        # Solve (K + lam*I) alpha = y using SVD for stability
        # K is symmetric positive semi-definite
        U, s, Vt = np.linalg.svd(K)
        # Inverse of (K + lam*I) is V * (1/(s+lam)) * U^T
        # But since K is symmetric, U=V
        # Let's just use the direct formula with SVD components
        # alpha = V @ ( (U^T @ y) / (s + lam) )
        Uy = U.T @ y
        alpha = Vt.T @ (Uy / (s + lam))
        return alpha

    for gamma in gammas:
        if _time_left() < 10.0:
            break
            
        # Precompute training kernel for this gamma
        K_tr = rbf_kernel(X_tr, X_tr, gamma)
        
        for lam in lambdas:
            if _time_left() < 10.0:
                break
                
            try:
                # Solve for alpha
                alpha = solve_ridge_svd(K_tr, y_tr, lam)
                
                # Predict on validation set
                K_val_tr = rbf_kernel(X_val, X_tr, gamma)
                pred_val = K_val_tr @ alpha
                
                err = np.mean((pred_val - y_val)**2)
                
                if err < best_val_error:
                    best_val_error = err
                    best_params = (gamma, lam)
            except Exception:
                continue

    # If no parameters found (should not happen), fallback to simple mean
    if best_params is None:
        def predict(X):
            return np.full(len(X), np.mean(y))
        return predict

    gamma_best, lam_best = best_params
    
    # Refit on full training set with best parameters
    K_full = rbf_kernel(x, x, gamma_best)
    alpha_full = solve_ridge_svd(K_full, y, lam_best)
    
    # Store necessary data for prediction
    X_train_scaled_store = x.copy()
    alpha_store = alpha_full.copy()
    gamma_store = gamma_best
    mu_store = mu
    sigma_store = sigma
    
    def predict(X):
        Xs = (X - mu_store) / sigma_store
        xv = Xs[:, 0]
        K_val = rbf_kernel(xv, X_train_scaled_store, gamma_store)
        return K_val @ alpha_store

    return predict
