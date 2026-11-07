import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, cholesky

def fit(X_train, y_train):
    # Flatten X for 1D processing, but keep general shape handling
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    # Normalize inputs to [0, 1] range to help kernel scaling
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    # Avoid division by zero
    x_range = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
    
    def normalize(X):
        return (X - x_min) / x_range
        
    X_norm = normalize(X_train)
    
    # Matern 5/2 Kernel
    def matern52_kernel(X1, X2, l, sigma_f):
        # X1: (n, d), X2: (m, d)
        # Compute Euclidean distances
        # For 1D, we can do this efficiently
        d = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=-1)
        # Matern 5/2: K = (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l) * sigma_f^2
        sqrt5 = np.sqrt(5.0)
        d_l = d / l
        k = (1.0 + sqrt5 * d_l + 5.0 * d_l**2 / 3.0) * np.exp(-sqrt5 * d_l)
        return k * sigma_f**2

    def log_marginal_likelihood(theta):
        l, log_sigma_f = theta
        sigma_f = np.exp(log_sigma_f)
        # Ensure length scale is positive
        l = np.exp(l)
        
        K = matern52_kernel(X_norm, X_norm, l, sigma_f)
        # Add small jitter for numerical stability
        K += 1e-6 * np.eye(len(y_train))
        
        try:
            L = cho_factor(K, lower=True)
            # Log likelihood: -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
            # det(K) = 2 * sum(log(diag(L)))
            alpha = cho_solve(L, y_train)
            log_det = 2.0 * np.sum(np.log(np.diag(cholesky(K, lower=True))))
            n = len(y_train)
            ll = -0.5 * np.dot(y_train, alpha) - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)
            return -ll # Negate for minimization
        except Exception:
            return 1e10

    # Initial guess for hyperparameters
    # Length scale l: typical scale of input is 1 after normalization, so l ~ 1.0
    # Signal variance sigma_f: standard deviation of y
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0
        
    init_l = 1.0
    init_sigma_f = y_std
    
    # Optimize log(l) and log(sigma_f) to keep them positive
    x0 = [np.log(init_l), np.log(init_sigma_f)]
    
    # Bounds for log parameters
    bounds = [(-2.0, 2.0), (-2.0, 2.0)] # l in [exp(-2), exp(2)] ~ [0.13, 7.39], sigma_f in [0.13, 7.39]
    
    res = minimize(log_marginal_likelihood, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
    
    opt_l = np.exp(res.x[0])
    opt_sigma_f = np.exp(res.x[1])
    
    # Precompute Cholesky factor for the training kernel with optimized hyperparams
    K_train = matern52_kernel(X_norm, X_norm, opt_l, opt_sigma_f)
    K_train += 1e-6 * np.eye(len(y_train))
    L_train = cho_factor(K_train, lower=True)
    alpha = cho_solve(L_train, y_train)
    
    def predict(X):
        X = np.asarray(X)
        X_norm_pred = normalize(X)
        
        # Compute K(X_test, X_train)
        K_test_train = matern52_kernel(X_norm_pred, X_norm, opt_l, opt_sigma_f)
        
        # Prediction mean: K_test_train @ K_train^-1 @ y_train
        # K_train^-1 @ y_train = alpha
        mu = K_test_train @ alpha
        
        # Return predictions
        return mu

    return predict
