import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 70  # 70s per equation, 4 equations = 280s, leaves 20s buffer
    
    X = X_train.reshape(-1, 1)
    y = y_train.reshape(-1)
    n = len(y)
    
    # Standardize
    X_mean = np.mean(X)
    X_std = np.std(X)
    if X_std < 1e-8:
        X_std = 1.0
    X_scaled = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-8:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    # Matérn 5/2 kernel
    def matern52_kernel(X1, X2, length_scale):
        dist = cdist(X1, X2)
        sqrt5 = np.sqrt(5)
        return np.exp(-sqrt5 * dist / length_scale) * (1 + sqrt5 * dist / length_scale)
    
    # Log marginal likelihood
    def log_marginal_likelihood(params):
        if time.time() > deadline - 5:
            return -np.inf
        log_var, log_length_scale = params
        var = np.exp(log_var)
        l = np.exp(log_length_scale)
        
        K = matern52_kernel(X_scaled, X_scaled, l) * var
        K += 1e-6 * np.eye(n) # Small noise for numerical stability
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
            lml = -0.5 * np.dot(y_scaled, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return -lml # Minimize negative log likelihood
        except np.linalg.LinAlgError:
            return 1e10
    
    # Optimize hyperparameters
    best_lml = 1e10
    best_params = [0.0, 0.0] # log_var=0, log_l=0
    
    # Grid search initial points + Nelder-Mead
    for init_log_var in [-1, 0, 1]:
        for init_log_l in [-1, 0, 1]:
            if time.time() > deadline - 10:
                break
            x0 = [init_log_var, init_log_l]
            res = minimize(log_marginal_likelihood, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.fun < best_lml:
                best_lml = res.fun
                best_params = res.x
    
    log_var, log_l = best_params
    var = np.exp(log_var)
    l = np.exp(log_l)
    
    # Final Kernel
    K = matern52_kernel(X_scaled, X_scaled, l) * var
    K += 1e-6 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    except np.linalg.LinAlgError:
        # Fallback to mean prediction if Cholesky fails
        def predict_fallback(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict_fallback

    def predict(X_test):
        X_test_scaled = (X_test.reshape(-1, 1) - X_mean) / X_std
        K_star = matern52_kernel(X_test_scaled, X_scaled, l) * var
        # Prediction mean
        mu = K_star @ alpha
        # Scale back
        return mu * y_std + y_mean

    return predict
