"""Gaussian Process regression with Matérn-5/2 kernel and NLML hyperparameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def fit(X_train, y_train):
    n = X_train.shape[0]
    x = X_train[:, 0].copy()
    y = y_train.copy()
    
    # Standardize x and y
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, signal_var):
        diff = x1[:, None] - x2[None, :]
        z = np.abs(diff) / length_scale
        kernel = (1.0 + np.sqrt(5.0) * z + (5.0 / 3.0) * z**2) * np.exp(-np.sqrt(5.0) * z)
        return signal_var * kernel

    def compute_nll(params):
        log_ls, log_sv, log_nv = params
        ls = np.exp(log_ls)
        sv = np.exp(log_sv)
        nv = np.exp(log_nv)
        
        K = matern52_kernel(x_scaled, x_scaled, ls, sv)
        K += nv * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        
        nll = 0.5 * y_scaled @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        
        return nll

    # Initial guesses in log space
    # ls ~ 1.0, sv ~ 1.0, nv ~ 1e-5
    x0 = np.array([0.0, 0.0, np.log(1e-5)])
    
    # Bounds in log space
    bounds = [(-4.6, 2.3), (-4.6, 2.3), (-18.4, -4.6)]
    
    result = minimize(compute_nll, x0, method='L-BFGS-B', bounds=bounds)
    
    if result.success or result.fun < 1e10:
        best_params = result.x
        best_ls = np.exp(best_params[0])
        best_sv = np.exp(best_params[1])
        best_nv = np.exp(best_params[2])
    else:
        # Fallback to defaults if optimization fails
        best_ls = 1.0
        best_sv = 1.0
        best_nv = 1e-5

    # Final fit
    K = matern52_kernel(x_scaled, x_scaled, best_ls, best_sv)
    K += best_nv * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
        
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    
    def predict(X):
        x_new = X[:, 0].copy()
        x_new_scaled = (x_new - x_mean) / x_std
        
        K_star = matern52_kernel(x_new_scaled, x_scaled, best_ls, best_sv)
        y_pred_scaled = K_star @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        return y_pred
        
    return predict
# EVOLVE-BLOCK-END
