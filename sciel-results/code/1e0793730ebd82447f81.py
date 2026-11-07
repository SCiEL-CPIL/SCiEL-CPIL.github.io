"""Gaussian Process regression with Matérn-5/2 kernel and optimized hyperparameters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    # X_train shape (n, d), y_train shape (n,)
    # All equations are 1D, so d=1
    n = X_train.shape[0]
    x = X_train[:, 0].copy()
    y = y_train.copy()
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Center y for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, signal_var):
        """Matérn 5/2 kernel for 1D inputs (already scaled)."""
        diff = x1[:, None] - x2[None, :]
        z = np.abs(diff) / length_scale
        # Matérn 5/2: (1 + sqrt(5)*z + 5/3*z^2) * exp(-sqrt(5)*z)
        kernel = (1.0 + np.sqrt(5.0) * z + (5.0 / 3.0) * z**2) * np.exp(-np.sqrt(5.0) * z)
        return signal_var * kernel
    
    def neg_log_likelihood(params, x_data, y_data):
        """Negative log marginal likelihood with jitter."""
        length_scale = np.exp(params[0])  # ensure positive
        signal_var = np.exp(params[1])    # ensure positive
        noise_var = np.exp(params[2])     # ensure positive
        
        K = matern52_kernel(x_data, x_data, length_scale, signal_var) + noise_var * np.eye(len(x_data))
        
        # Add jitter for numerical stability
        jitter = 1e-6 * np.eye(len(x_data))
        K = K + jitter
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10  # very large penalty
        
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_data))
        nll = 0.5 * y_data @ alpha_vec + np.sum(np.log(np.diag(L))) + 0.5 * len(y_data) * np.log(2 * np.pi)
        return nll
    
    # Optimize hyperparameters using BFGS
    # Initial guess: length_scale=1.0, signal_var=1.0, noise_var=0.01
    # In log space: log(1.0)=0, log(1.0)=0, log(0.01)=-4.6
    x0 = np.array([0.0, 0.0, -4.6])
    
    result = minimize(
        neg_log_likelihood,
        x0,
        args=(x_scaled, y_scaled),
        method='L-BFGS-B',
        bounds=[(-5, 5), (-5, 5), (-10, 0)],
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    # Extract optimized hyperparameters
    best_ls = np.exp(result.x[0])
    best_sv = np.exp(result.x[1])
    best_nv = np.exp(result.x[2])
    
    # Final GP fit using all training data with best hyperparameters
    K = matern52_kernel(x_scaled, x_scaled, best_ls, best_sv) + best_nv * np.eye(n)
    
    # Add jitter for numerical stability
    K = K + 1e-6 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-4 * np.eye(n)
        L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    
    # Create predict function
    def predict(X):
        x_new = X[:, 0].copy()
        x_new_scaled = (x_new - x_mean) / x_std
        
        K_star = matern52_kernel(x_new_scaled, x_scaled, best_ls, best_sv)
        y_pred_scaled = K_star @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
