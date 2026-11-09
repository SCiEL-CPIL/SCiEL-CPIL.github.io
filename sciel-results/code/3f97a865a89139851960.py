# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 305.0  # 25s margin for 330s kill

    n, d = X_train.shape
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    y_var = float(np.var(y_train))

    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    def check_time():
        return time.time() < deadline

    # Normalize inputs to [0, 1] range for better kernel scaling
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0

    def normalize(X):
        return (X - X_min) / X_range

    Xn = normalize(X_train).astype(np.float64)
    yn = (y_train - y_mean) / (y_std + 1e-12)

    # Gaussian Process Regression with Composite Kernel
    # Kernel: RBF + Linear + White Noise
    # We optimize log-marginal likelihood
    
    def log_marginal_likelihood(params):
        if not check_time():
            return -np.inf
            
        # params: [log_sigma_f, log_l, log_sigma_lin, log_sigma_n]
        try:
            log_sigma_f, log_l, log_sigma_lin, log_sigma_n = params
            sigma_f = np.exp(log_sigma_f)
            l = np.exp(log_l)
            sigma_lin = np.exp(log_sigma_lin)
            sigma_n = np.exp(log_sigma_n)
            
            # Compute RBF kernel matrix
            # K_rbf[i,j] = sigma_f^2 * exp(-||x_i - x_j||^2 / (2*l^2))
            dists = cdist(Xn, Xn, metric='sqeuclidean')
            K_rbf = (sigma_f ** 2) * np.exp(-dists / (2.0 * l ** 2))
            
            # Compute Linear kernel matrix
            # K_lin[i,j] = sigma_lin^2 * (x_i . x_j)
            K_lin = (sigma_lin ** 2) * (Xn @ Xn.T)
            
            # Total Kernel + Noise
            K = K_rbf + K_lin
            K[np.diag_indices_from(K)] += (sigma_n ** 2)
            
            # Solve K alpha = y
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                return -np.inf
                
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            
            # Log marginal likelihood
            # 0.5 * y^T K^-1 y - 0.5 * log(det(K)) - N/2 * log(2*pi)
            det_log = 2.0 * np.sum(np.log(np.diag(L)))
            data_fit = yn @ alpha
            lml = -0.5 * data_fit - 0.5 * det_log - 0.5 * n * np.log(2.0 * np.pi)
            
            return -lml # Minimize negative LML
            
        except Exception:
            return -np.inf

    # Initial guesses for hyperparameters
    # sigma_f ~ 1, l ~ 1, sigma_lin ~ 1, sigma_n ~ 0.01
    init_params = np.log(np.array([1.0, 1.0, 1.0, 0.01]))
    
    best_lml = -np.inf
    best_params = init_params
    best_alpha = None
    best_L = None
    best_K_diag = None

    # Try a few different initializations or bounds if needed, but one good run is often enough for GPR
    # We can use L-BFGS-B
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-10, 2)] # Log space bounds
    
    res = minimize(log_marginal_likelihood, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
    
    if check_time():
        if res.fun < best_lml or best_lml == -np.inf:
            best_lml = res.fun
            best_params = res.x
            
            # Reconstruct optimal kernel and solve for alpha
            log_sigma_f, log_l, log_sigma_lin, log_sigma_n = best_params
            sigma_f = np.exp(log_sigma_f)
            l = np.exp(log_l)
            sigma_lin = np.exp(log_sigma_lin)
            sigma_n = np.exp(log_sigma_n)
            
            dists = cdist(Xn, Xn, metric='sqeuclidean')
            K_rbf = (sigma_f ** 2) * np.exp(-dists / (2.0 * l ** 2))
            K_lin = (sigma_lin ** 2) * (Xn @ Xn.T)
            K = K_rbf + K_lin
            K[np.diag_indices_from(K)] += (sigma_n ** 2)
            
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
                best_alpha = alpha
                best_L = L
            except Exception:
                pass

    # If optimization failed or didn't improve, use initial params or a simple fallback
    if best_alpha is None:
        # Fallback to simple linear regression if GPR fails
        try:
            X_aug = np.column_stack([X_train, np.ones(n)])
            w, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
            def predict(X):
                X_aug_test = np.column_stack([X, np.ones(X.shape[0])])
                return X_aug_test @ w
            return predict
        except Exception:
            return lambda X: np.full(X.shape[0], y_mean)

    # Define prediction function
    def predict(X):
        Xn_test = normalize(X).astype(np.float64)
        
        log_sigma_f, log_l, log_sigma_lin, log_sigma_n = best_params
        sigma_f = np.exp(log_sigma_f)
        l = np.exp(log_l)
        sigma_lin = np.exp(log_sigma_lin)
        
        # Compute cross kernel K(x_test, X_train)
        dists_test = cdist(Xn_test, Xn, metric='sqeuclidean')
        K_rbf_test = (sigma_f ** 2) * np.exp(-dists_test / (2.0 * l ** 2))
        K_lin_test = (sigma_lin ** 2) * (Xn_test @ Xn.T)
        
        K_star = K_rbf_test + K_lin_test
        
        # Prediction mean
        mu = K_star @ best_alpha
        
        # Denormalize
        return (mu * y_std) + y_mean

    # Validate
    try:
        preds = predict(X_train)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
             # Fallback
             X_aug = np.column_stack([X_train, np.ones(n)])
             w, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
             return lambda X: np.column_stack([X, np.ones(X.shape[0])]) @ w
        return predict
    except Exception:
        return lambda X: np.full(X.shape[0], y_mean)

# EVOLVE-BLOCK-END
