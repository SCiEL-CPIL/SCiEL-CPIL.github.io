import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n_samples, n_features = X_train.shape
    
    # Standardize inputs
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_train_std = (X_train - X_mean) / X_std
    
    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_var = 1.0
    y_std = np.sqrt(y_var)
    
    # Pre-compute Gram matrix for Poly kernel: G = X_std @ X_std.T
    G = X_train_std @ X_train_std.T
    # Pre-compute dist_sq for RBF
    X_sq_norms = np.sum(X_train_std ** 2, axis=1)
    dist_sq = X_sq_norms[:, None] + X_sq_norms[None, :] - 2.0 * G
    np.maximum(dist_sq, 0, out=dist_sq)
    
    # Kernel components
    def compute_kernel(theta):
        log_sigma_f, log_l, log_sigma_p, log_sigma_n = theta
        sigma_f = np.exp(log_sigma_f)
        l = np.exp(log_l)
        sigma_p = np.exp(log_sigma_p)
        sigma_n = np.exp(log_sigma_n)
        
        K_rbf = sigma_f ** 2 * np.exp(-0.5 * dist_sq / (l ** 2))
        # Degree 2 Polynomial kernel
        K_poly = sigma_p ** 2 * (G + 1.0) ** 2
        
        K = K_rbf + K_poly + sigma_n ** 2 * np.eye(n_samples)
        return K

    def neg_log_likelihood(theta):
        try:
            K = compute_kernel(theta)
            K += 1e-8 * np.eye(n_samples)
            
            L, _ = cho_factor(K)
            alpha, _ = cho_solve((L, True), y_centered)
            
            log_det = 2 * np.sum(np.log(np.diag(L)))
            y_alpha = y_centered @ alpha
            
            nll = 0.5 * y_alpha + 0.5 * log_det + 0.5 * n_samples * np.log(2 * np.pi)
            return nll
        except Exception:
            return 1e10

    # Initial hyperparameters
    theta0 = [np.log(y_std), 0.0, np.log(y_std), np.log(0.1)]
    bounds = [(-2, 5), (-2, 3), (-2, 5), (-5, 2)]
    
    best_nll = np.inf
    best_theta = theta0
    
    # Optimization
    try:
        res = minimize(neg_log_likelihood, theta0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-6})
        if res.success and res.fun < best_nll:
            best_nll = res.fun
            best_theta = res.x
    except Exception:
        pass
        
    try:
        res2 = minimize(neg_log_likelihood, theta0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if res2.success and res2.fun < best_nll:
            best_nll = res2.fun
            best_theta = res2.x
    except Exception:
        pass

    # Final Kernel and Solve
    K_final = compute_kernel(best_theta)
    K_final += 1e-8 * np.eye(n_samples)
    L_final, _ = cho_factor(K_final)
    alpha_final, _ = cho_solve((L_final, True), y_centered)
    
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_pred_std = (X - X_mean) / X_std
        
        # RBF part
        X_pred_sq = np.sum(X_pred_std ** 2, axis=1)
        X_train_sq = np.sum(X_train_std ** 2, axis=1)
        dist_pred_sq = X_pred_sq[:, None] + X_train_sq[None, :] - 2.0 * (X_pred_std @ X_train_std.T)
        np.maximum(dist_pred_sq, 0, out=dist_pred_sq)
        
        log_sigma_f, log_l, log_sigma_p, log_sigma_n = best_theta
        sigma_f = np.exp(log_sigma_f)
        l = np.exp(log_l)
        sigma_p = np.exp(log_sigma_p)
        
        K_pred_rbf = sigma_f ** 2 * np.exp(-0.5 * dist_pred_sq / (l ** 2))
        
        # Poly part
        G_pred = X_pred_std @ X_train_std.T
        K_pred_poly = sigma_p ** 2 * (G_pred + 1.0) ** 2
        
        K_pred = K_pred_rbf + K_pred_poly
        
        y_pred = K_pred @ alpha_final + y_mean
        
        return y_pred

    return predict
