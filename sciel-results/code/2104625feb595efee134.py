"""
Strategy: Gaussian Process Regression with Matérn 5/2 Kernel.
Instead of piecewise splines, we use a GP which provides a principled way to
interpolate noiseless data while controlling smoothness via the length scale.
The Matérn 5/2 kernel is chosen for its flexibility (smooth but not infinitely
differentiable like RBF), which is well-suited for physical or symbolic functions.
We optimize the length scale via a simple grid search on log-spaced values.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    n = len(X)
    
    # Normalize y for numerical stability in GP calculations
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Sort data for potential speedups or just consistency
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted_norm = y_norm[sort_idx]
    
    best_predict = None
    best_score = np.inf
    best_len_scale = 1.0
    
    # Define Matern 5/2 kernel
    def matern52_kernel(x1, x2, l):
        r = np.abs(x1 - x2) / l
        return (1 + np.sqrt(5) * r + (5/3) * r**2) * np.exp(-np.sqrt(5) * r)
    
    # Precompute kernel matrix for training data for each candidate length scale?
    # No, we need to evaluate likelihood for each.
    # With n=200, K is 200x200. Cholesky is O(n^3/3) ~ 2.6M ops. Very fast.
    
    # We want to maximize the log marginal likelihood (or minimize negative log marginal likelihood).
    # NLL = 0.5 * y^T K^-1 y + 0.5 * log(det(K)) + n/2 * log(2*pi)
    
    def nll_log_l(log_l):
        if time.time() > deadline:
            return 1e10
        l = np.exp(log_l)
        K = matern52_kernel(X_sorted[:, None], X_sorted[None, :], l)
        # Add small jitter for numerical stability
        K += 1e-8 * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            # log(det(K)) = 2 * sum(log(diag(L)))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve L v = y
            v = np.linalg.solve(L, y_sorted_norm)
            # Quadratic form: y^T K^-1 y = v^T v
            quad_form = np.dot(v, v)
            
            nll = 0.5 * quad_form + 0.5 * log_det
            return nll
        except np.linalg.LinAlgError:
            return 1e10
    
    # Grid search over length scales
    # Typical ranges for inputs in [-3,3] or [0.1,6]
    # Log space from 0.1 to 10
    log_l_grid = np.linspace(np.log(0.1), np.log(10.0), 50)
    
    best_nll = np.inf
    best_log_l = np.log(1.0)
    
    for log_l in log_l_grid:
        if time.time() > deadline:
            break
        val = nll_log_l(log_l)
        if val < best_nll:
            best_nll = val
            best_log_l = log_l
    
    # Refine around best grid point
    if time.time() < deadline - 5:
        try:
            res = minimize_scalar(nll_log_l, bounds=(best_log_l - 0.5, best_log_l + 0.5), method='bounded', options={'xatol': 1e-4})
            if res.success:
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_log_l = res.x
        except Exception:
            pass
    
    best_len_scale = np.exp(best_log_l)
    
    # Final GP fit with best length scale
    K = matern52_kernel(X_sorted[:, None], X_sorted[None, :], best_len_scale)
    K += 1e-8 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        # Precompute inverse Cholesky for prediction
        # K^-1 = L^-T L^-1
        # We need alpha = K^-1 y for prediction: f(x) = k(x)^T K^-1 y
        alpha = np.linalg.solve(L, y_sorted_norm)
        alpha = np.linalg.solve(L.T, alpha)
        
        def predict(X):
            if time.time() > deadline:
                 # Fallback if we run out of time during prediction (unlikely but safe)
                 return np.full(X.shape[0], y_mean)
            
            X_flat = X.flatten()
            
            # Compute kernel between test points and training points
            # k(x_test, x_train) shape (m, n)
            K_s = matern52_kernel(X_flat[:, None], X_sorted[None, :], best_len_scale)
            
            # Predicted mean in normalized space
            y_pred_norm = K_s @ alpha
            
            # Transform back to original scale
            y_pred = y_pred_norm * y_std + y_mean
            
            return y_pred.reshape(X.shape)
        
        best_predict = predict
        
    except Exception:
        # Fallback if GP fails
        # Use simple linear interpolation or mean
        def predict_fallback(X):
            X_flat = X.flatten()
            # Nearest neighbor or linear interp
            # For simplicity, use np.interp
            y_pred = np.interp(X_flat, X_sorted, y)
            return y_pred.reshape(X.shape)
        best_predict = predict_fallback

    return best_predict
# EVOLVE-BLOCK-END
