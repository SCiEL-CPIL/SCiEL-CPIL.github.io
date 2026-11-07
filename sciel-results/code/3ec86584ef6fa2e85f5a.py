"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Standardize inputs to have zero mean and unit variance
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    
    def standardize(X):
        return (X - X_mean) / X_std
    
    Xn_train = standardize(X_train)
    
    # Matern 2.5 Kernel
    # k(x, x') = (1 + sqrt(5)*r + (5/3)*r^2) * exp(-sqrt(5)*r)
    # where r is the anisotropic Euclidean distance: sqrt(sum((xi-xj)^2 / l_i^2))
    def matern_25_kernel(X1, X2, lengthscales):
        # X1: (n, d), X2: (m, d)
        # lengthscales: (d,)
        
        X1_s = X1 / lengthscales
        X2_s = X2 / lengthscales
        
        # Squared Euclidean distance
        x1_sq = np.sum(X1_s ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(X2_s ** 2, axis=1, keepdims=True)
        dist_sq = x1_sq + x2_sq.T - 2 * (X1_s @ X2_s.T)
        np.maximum(dist_sq, 0, out=dist_sq) # numerical stability
        dist = np.sqrt(dist_sq)
        
        sqrt5 = np.sqrt(5.0)
        return (1.0 + sqrt5 * dist + (5.0/3.0) * dist**2) * np.exp(-sqrt5 * dist)

    # Hyperparameters: log(lengthscales) for each dim, log(signal_var)
    # We fix noise_var to a very small value to enforce interpolation (noiseless data)
    
    def log_marginal_likelihood(params):
        # params: [log_l1, ..., log_ld, log_sigma_f]
        log_ls = params[:d]
        log_sf = params[d]
        
        lengthscales = np.exp(log_ls)
        sigma_f = np.exp(log_sf)
        noise_var = 1e-10 # Fixed small noise
        
        K = matern_25_kernel(Xn_train, Xn_train, lengthscales)
        K += noise_var * np.eye(n)
        
        # Add small jitter for numerical stability
        K += 1e-12 * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10 # Large penalty
        
        # Compute alpha = K^-1 y
        # L L^T alpha = y  => L^T alpha = L^-1 y
        v = np.linalg.solve(L, y_train)
        
        # NLML = 0.5 * y^T K^-1 y + 0.5 * log det K + 0.5 * n * log(2*pi)
        # log det K = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        
        nlml = 0.5 * np.dot(y_train, v) + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        
        return nlml

    # Initial guess: length scales = 1, signal var = var(y)
    init_ls = np.ones(d)
    init_sf = np.var(y_train)
    if init_sf < 1e-8:
        init_sf = 1.0
        
    x0 = np.concatenate([np.log(init_ls), [np.log(init_sf)]])
    
    # Optimize using Nelder-Mead for robustness in low dimensions
    result = minimize(log_marginal_likelihood, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    log_ls = best_params[:d]
    log_sf = best_params[d]
    
    lengthscales = np.exp(log_ls)
    sigma_f = np.exp(log_sf)
    noise_var = 1e-10
    
    # Final fit
    K = matern_25_kernel(Xn_train, Xn_train, lengthscales)
    K += noise_var * np.eye(n)
    K += 1e-12 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        v = np.linalg.solve(L, y_train)
        alpha = np.linalg.solve(L.T, v)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
        
    def predict(X):
        Xn = standardize(X)
        K_star = matern_25_kernel(Xn, Xn_train, lengthscales)
        return K_star @ alpha

    return predict
# EVOLVE-BLOCK-END
