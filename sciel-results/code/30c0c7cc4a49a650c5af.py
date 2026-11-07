"""Symbolic regression over 10 hidden equations: RBF kernel regression with adaptive bandwidth."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import cdist

_START = time.time()
_DEADLINE = 295.0  # Safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _rbf_kernel(X, Y, sigma):
    """Compute RBF kernel matrix between X and Y."""
    # Use squared Euclidean distance
    d2 = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-d2 / (2.0 * sigma * sigma))

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    N, D = X.shape
    
    # Emergency fallback
    if _remaining() < 5.0:
        mean_y = np.mean(y)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.ones(len(X)) * mean_y
        return predict
    
    # Compute median distance for initial sigma estimate
    if N > 1:
        dists = cdist(X, X, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        median_dist = np.median(dists)
        if median_dist > 0:
            sigma_init = median_dist / 2.0
        else:
            sigma_init = 1.0
    else:
        sigma_init = 1.0
    
    # Try multiple sigma values
    sigma_candidates = [sigma_init * 0.1, sigma_init * 0.3, sigma_init * 0.5, sigma_init, 
                       sigma_init * 2.0, sigma_init * 5.0, sigma_init * 10.0]
    
    best_sigma = sigma_init
    best_alpha = None
    best_train_mse = np.inf
    
    for sigma in sigma_candidates:
        if _remaining() < 3.0:
            break
        
        try:
            K = _rbf_kernel(X, X, sigma)
            # Add small ridge for numerical stability
            K_reg = K + 1e-6 * np.eye(N)
            
            # Solve for alpha
            alpha = np.linalg.solve(K_reg, y)
            
            # Compute training MSE
            pred = K @ alpha
            mse = np.mean((y - pred) ** 2)
            
            if mse < best_train_mse:
                best_train_mse = mse
                best_sigma = sigma
                best_alpha = alpha
        except Exception:
            continue
    
    # If RBF didn't work well, try linear fallback
    if best_alpha is None or best_train_mse > np.var(y) * 0.1:
        # Try linear regression
        try:
            A_lin = np.column_stack([np.ones(N), X])
            w, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
            pred_lin = A_lin @ w
            mse_lin = np.mean((y - pred_lin) ** 2)
            
            # Also try quadratic
            if D <= 3:
                cols_quad = [np.ones(N)]
                for j in range(D):
                    x = X[:, j]
                    cols_quad.append(x)
                    cols_quad.append(x**2)
                # Interactions
                for i in range(D):
                    for j in range(i+1, D):
                        cols_quad.append(X[:, i] * X[:, j])
                A_quad = np.column_stack(cols_quad)
                w_quad, *_ = np.linalg.lstsq(A_quad, y, rcond=None)
                pred_quad = A_quad @ w_quad
                mse_quad = np.mean((y - pred_quad) ** 2)
                
                if mse_quad < mse_lin:
                    mse_lin = mse_quad
                    w = w_quad
                    A_lin = A_quad
                    use_linear = True
                else:
                    use_linear = True
            else:
                use_linear = True
            
            if use_linear and mse_lin < best_train_mse:
                # Use linear model instead
                def predict(X):
                    X = np.asarray(X, dtype=float)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    N_test, D_test = X.shape
                    A_test = np.column_stack([np.ones(N_test), X])
                    if A_test.shape[1] != len(w):
                        # Fallback to mean
                        return np.ones(N_test) * np.mean(y)
                    return A_test @ w
                return predict
        except Exception:
            pass
    
    # Create RBF predict function
    if best_alpha is not None:
        X_train_copy = X.copy()
        alpha_copy = best_alpha.copy()
        sigma_copy = best_sigma
        y_mean = np.mean(y)
        
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            N_test = len(X)
            try:
                K_test = _rbf_kernel(X, X_train_copy, sigma_copy)
                result = K_test @ alpha_copy
                if not np.all(np.isfinite(result)):
                    result = np.where(np.isfinite(result), result, y_mean)
                return result
            except Exception:
                return np.ones(N_test) * y_mean
        return predict
    else:
        mean_y = np.mean(y)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.ones(len(X)) * mean_y
        return predict
# EVOLVE-BLOCK-END
