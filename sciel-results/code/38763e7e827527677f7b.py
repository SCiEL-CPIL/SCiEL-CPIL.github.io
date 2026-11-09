import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    """
    Fits a regression model using a Gaussian Process with an anisotropic RBF kernel.
    Hyperparameters (length scales, signal variance, noise variance) are optimized
    via maximum likelihood estimation (minimizing negative log marginal likelihood).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs to have zero mean and unit variance
    # This helps in optimizing length scales as they become comparable
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Standardize y for numerical stability in GP calculations?
    # Usually not strictly necessary for GP, but helps with conditioning if y scale is huge/tiny.
    # Let's stick to raw y for simplicity, GP handles scale via sigma.
    
    # Initial hyperparameters
    # log_alpha = log(signal_variance)
    # log_l = log(length_scales)
    # log_sigma_n = log(noise_variance)
    
    # Heuristic initial guesses
    # Signal variance: roughly var(y)
    y_var = np.var(y)
    if y_var < 1e-8:
        y_var = 1.0
    init_log_alpha = np.log(y_var)
    
    # Length scales: start with 1.0 for all dimensions
    init_log_l = np.zeros(d)
    
    # Noise: start small
    init_log_sigma_n = np.log(1e-4)
    
    def neg_log_likelihood(params):
        if time.time() > DEADLINE:
            return 1e100 # Timeout penalty
        
        log_alpha = params[0]
        log_l = params[1:1+d]
        log_sigma_n = params[1+d]
        
        # Clip parameters to avoid numerical issues
        # alpha in [1e-6, 1e6]
        alpha = np.exp(np.clip(log_alpha, -13.8, 13.8))
        # l in [0.01, 10.0]
        l = np.exp(np.clip(log_l, -4.6, 2.3))
        # sigma_n in [1e-8, 1.0]
        sigma_n = np.exp(np.clip(log_sigma_n, -18.4, 0.0))
        
        # Construct Kernel Matrix
        # K(x, x') = alpha * exp(-0.5 * sum((xi - xj)^2 / l_i^2)) + sigma_n^2 * I
        
        # Calculate squared distances efficiently
        # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x . x'
        # But for anisotropic, we need element-wise distances.
        
        # Reshape X_norm to (n, 1, d) and (1, n, d)
        X_col = X_norm[:, np.newaxis, :]
        X_row = X_norm[np.newaxis, :, :]
        
        # Squared differences: (n, n, d)
        diff = X_col - X_row
        
        # Weighted squared distances: (n, n)
        # (diff^2 / l^2).sum(axis=2)
        # l is shape (d,), broadcast to (1, 1, d)
        l_sq = l**2
        dist_sq = np.sum((diff**2) / l_sq, axis=2)
        
        # Kernel
        K = alpha * np.exp(-0.5 * dist_sq)
        K[np.diag_indices_from(K)] += sigma_n**2
        
        # Compute negative log marginal likelihood
        # NLL = 0.5 * y^T K^{-1} y + 0.5 * log(det(K)) + n/2 * log(2*pi)
        
        try:
            # Use Cholesky decomposition for stability
            L = np.linalg.cholesky(K)
            # log(det(K)) = 2 * sum(log(diag(L)))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            
            # Solve L * v = y -> v
            v = cho_solve((L, True), y)
            # y^T K^{-1} y = v^T v
            yK_inv_y = np.dot(v, v)
            
            nll = 0.5 * yK_inv_y + 0.5 * log_det_K + 0.5 * n * np.log(2.0 * np.pi)
            
            return nll
        except np.linalg.LinAlgError:
            return 1e100

    # Optimization
    x0 = np.concatenate(([init_log_alpha], init_log_l, [init_log_sigma_n]))
    
    best_params = x0
    best_nll = neg_log_likelihood(x0)
    
    try:
        # Use L-BFGS-B as it handles bounds well and is efficient
        res = minimize(
            neg_log_likelihood, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 200}
        )
        
        if res.success or res.fun < best_nll:
            best_params = res.x
            best_nll = res.fun
    except Exception:
        pass
    
    # Extract best hyperparameters
    log_alpha = best_params[0]
    log_l = best_params[1:1+d]
    log_sigma_n = best_params[1+d]
    
    alpha = np.exp(np.clip(log_alpha, -13.8, 13.8))
    l = np.exp(np.clip(log_l, -4.6, 2.3))
    sigma_n = np.exp(np.clip(log_sigma_n, -18.4, 0.0))
    
    # Precompute Cholesky factor for training data for prediction
    # We need K and K_inv for prediction: mu = K(X_test, X_train) K(X_train, X_train)^{-1} y
    # We can reuse the Cholesky factor from the last evaluation or recompute.
    # Let's recompute to be safe with the final best_params.
    
    X_col = X_norm[:, np.newaxis, :]
    X_row = X_norm[np.newaxis, :, :]
    diff = X_col - X_row
    l_sq = l**2
    dist_sq = np.sum((diff**2) / l_sq, axis=2)
    
    K = alpha * np.exp(-0.5 * dist_sq)
    K[np.diag_indices_from(K)] += sigma_n**2
    
    try:
        L = np.linalg.cholesky(K)
        # Precompute K_inv_y = K^{-1} y
        # K^{-1} y = L^{-T} L^{-1} y
        v = cho_solve((L, True), y)
        K_inv_y = v
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails (shouldn't happen with sigma_n > 0)
        # Use pseudo-inverse or simple linear regression
        try:
            K_inv_y = np.linalg.lstsq(K, y, rcond=None)[0]
        except:
            K_inv_y = np.zeros(n)
            alpha = 0.0
            l = np.ones(d)
            sigma_n = 1.0

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        n_test = X_test.shape[0]
        
        # Normalize test data
        X_test_norm = (X_test - X_mean) / X_std
        
        # Calculate cross kernel K(X_test, X_train)
        # Shape: (n_test, n_train)
        
        # If alpha is 0, return mean
        if alpha < 1e-10:
            return np.full(n_test, np.mean(y))
            
        X_test_col = X_test_norm[:, np.newaxis, :]
        X_train_row = X_norm[np.newaxis, :, :]
        
        diff = X_test_col - X_train_row
        dist_sq = np.sum((diff**2) / l_sq, axis=2)
        
        K_s = alpha * np.exp(-0.5 * dist_sq)
        
        # Predictions: mu = K_s @ K_inv_y
        mu = K_s @ K_inv_y
        
        return mu

    return predict
