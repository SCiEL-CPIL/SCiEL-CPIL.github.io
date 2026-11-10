"""Gaussian Process symbolic-regression fit over hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


def _matern52_kernel(X1, X2, lengthscales, signal_var, noise_var):
    """Compute Matern 5/2 kernel matrix with per-dimension anisotropic lengthscales."""
    # Standardize inputs per dimension using training stats (passed via closure or computed)
    # For flexibility, we compute distances on scaled inputs
    d = X1.shape[1]
    # Scale each dimension
    X1_scaled = X1 / lengthscales
    X2_scaled = X2 / lengthscales
    # Euclidean distance
    diff = X1_scaled[:, np.newaxis, :] - X2_scaled[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    # Matern 5/2
    sqrt3r = np.sqrt(3.0) * r
    K = signal_var * (1.0 + sqrt3r + (sqrt3r ** 2) / 3.0) * np.exp(-sqrt3r)
    return K


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Initial hyperparameters
    # Lengthscales: initialize to range of each dimension
    lengthscales = np.array([X[:, j].max() - X[:, j].min() for j in range(d)])
    lengthscales[lengthscales < 1e-8] = 1.0
    signal_var = np.var(y) + 1e-6
    noise_var = 1e-6  # minimal noise for noiseless data
    
    # Precompute scaled X for faster kernel computation
    def compute_log_marginal_likelihood(theta):
        if time.time() > deadline:
            return np.inf
        ls = np.exp(theta[:d])
        sv = np.exp(theta[d])
        nv = np.exp(theta[d + 1])
        
        X_scaled = X / ls
        diff = X_scaled[:, np.newaxis, :] - X_scaled[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        sqrt3r = np.sqrt(3.0) * r
        K = sv * (1.0 + sqrt3r + (sqrt3r ** 2) / 3.0) * np.exp(-sqrt3r)
        K[np.diag_indices_from(K)] += nv
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            lml = -0.5 * (y @ alpha + log_det) - 0.5 * n * np.log(2.0 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return np.inf
    
    # Optimize hyperparameters
    theta0 = np.concatenate([np.log(lengthscales), [np.log(signal_var), np.log(noise_var)]])
    
    best_theta = theta0
    best_lml = np.inf
    
    # Try a few random restarts
    n_restarts = min(5, max(1, int((deadline - time.time()) / 5.0)))
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            theta_start = theta0
        else:
            theta_start = theta0 + np.random.randn(len(theta0)) * 0.5
        
        res = minimize(compute_log_marginal_likelihood, theta_start, 
                      method='L-BFGS-B', 
                      options={'maxiter': 100, 'maxfun': 500})
        
        if res.fun < best_lml:
            best_lml = res.fun
            best_theta = res.x
    
    # Use best hyperparameters
    ls = np.exp(best_theta[:d])
    sv = np.exp(best_theta[d])
    nv = np.exp(best_theta[d + 1])
    
    # Compute Cholesky factor for prediction
    X_scaled = X / ls
    diff = X_scaled[:, np.newaxis, :] - X_scaled[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    sqrt3r = np.sqrt(3.0) * r
    K = sv * (1.0 + sqrt3r + (sqrt3r ** 2) / 3.0) * np.exp(-sqrt3r)
    K[np.diag_indices_from(K)] += nv
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback to ridge regression if GP fails
        X_design = np.column_stack([np.ones(n), X, X**2])
        w, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            X_t_design = np.column_stack([np.ones(len(X_t)), X_t, X_t**2])
            return X_t_design @ w
        return predict
    
    # Prediction function
    def predict(X_test):
        X_t = np.asarray(X_test, dtype=np.float64)
        X_t_scaled = X_t / ls
        diff = X_t_scaled[:, np.newaxis, :] - X_scaled[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        sqrt3r = np.sqrt(3.0) * r
        K_star = sv * (1.0 + sqrt3r + (sqrt3r ** 2) / 3.0) * np.exp(-sqrt3r)
        
        # Mean prediction
        mean = K_star @ alpha
        
        # Ensure finite output
        mean = np.where(np.isfinite(mean), mean, 0.0)
        
        return mean
    
    return predict
# EVOLVE-BLOCK-END
