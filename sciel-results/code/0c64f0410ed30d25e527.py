# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 305.0

    n, d = X_train.shape
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    y_var = float(np.var(y_train))

    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    def check_time():
        return time.time() < deadline

    # Normalize inputs to [0, 1]
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0

    def normalize(X):
        return (X - X_min) / X_range

    Xn = normalize(X_train).astype(np.float64)
    yn = (y_train - y_mean) / (y_std + 1e-12)

    # Gaussian Process Regression with Composite Kernel
    # We'll optimize hyperparameters: length_scale (for RBF), sigma_f (signal std), sigma_n (noise std)
    # Kernel: K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2) + sigma_n^2 * I

    # Precompute distance matrix for efficiency
    # K(x, x') = exp(-0.5 * sum_j (x_j - x'_j)^2 / l^2)
    # We can compute this as: exp(-0.5/l^2 * (||x||^2 + ||x'||^2 - 2*x.x'))
    
    Xn_sq = np.sum(Xn ** 2, axis=1)  # shape (n,)
    
    def compute_K(l):
        """Compute RBF kernel matrix given length scale l."""
        # Squared distances: d_ij^2 = ||xi||^2 + ||xj||^2 - 2*xi.xj
        sq_dists = Xn_sq[:, None] + Xn_sq[None, :] - 2.0 * (Xn @ Xn.T)
        # Clip to avoid negative values due to numerical errors
        sq_dists = np.maximum(sq_dists, 0.0)
        # RBF: exp(-0.5 * sq_dists / l^2)
        K = np.exp(-0.5 * sq_dists / (l * l))
        return K

    def log_marginal_likelihood(params):
        """Negative log marginal likelihood for optimization."""
        if not check_time():
            return 1e10
        
        l, log_sigma_f, log_sigma_n = params
        l = np.exp(l)  # ensure positive
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        # Clamp to reasonable ranges
        l = np.clip(l, 0.01, 10.0)
        sigma_f = np.clip(sigma_f, 0.001, 100.0)
        sigma_n = np.clip(sigma_n, 1e-6, 1.0)
        
        K = compute_K(l)
        K += sigma_n ** 2 * np.eye(n)
        
        # Compute log marginal likelihood
        # LML = -0.5 * y^T (K + sigma_n^2 I)^-1 y - 0.5 * log(det(K + sigma_n^2 I)) - n/2 * log(2*pi)
        # We optimize -LML
        
        try:
            # Cholesky decomposition for stability
            L = np.linalg.cholesky(K)
            # Solve L * alpha = y
            alpha = np.linalg.solve(L, yn)
            # alpha^T alpha
            alpha_sq = np.dot(alpha, alpha)
            # log det K = 2 * sum(log(diag(L)))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            
            lml = -0.5 * alpha_sq - 0.5 * log_det_K - n / 2.0 * np.log(2.0 * np.pi)
            return -lml  # minimize negative LML
        except np.linalg.LinAlgError:
            return 1e10

    # Optimize hyperparameters
    best_neg_lml = np.inf
    best_params = None
    best_K_inv_y = None
    best_chol = None
    
    # Multi-start optimization
    initial_guesses = [
        [0.0, 0.0, -5.0],   # l=1, sigma_f=1, sigma_n~0.007
        [0.5, 0.0, -3.0],   # l~1.65, sigma_f=1, sigma_n~0.05
        [-0.5, 0.0, -2.0],  # l~0.6, sigma_f=1, sigma_n~0.14
        [0.0, 0.5, -4.0],   # l=1, sigma_f~1.65, sigma_n~0.018
        [0.3, -0.5, -3.0],  # l~1.35, sigma_f~0.6, sigma_n~0.05
    ]
    
    for guess in initial_guesses:
        if not check_time():
            break
        try:
            res = minimize(log_marginal_likelihood, guess, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res.fun < best_neg_lml:
                best_neg_lml = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.0, 0.0, -5.0])
    
    l_opt = np.exp(best_params[0])
    l_opt = np.clip(l_opt, 0.01, 10.0)
    sigma_f_opt = np.exp(best_params[1])
    sigma_f_opt = np.clip(sigma_f_opt, 0.001, 100.0)
    sigma_n_opt = np.exp(best_params[2])
    sigma_n_opt = np.clip(sigma_n_opt, 1e-6, 1.0)
    
    # Compute final kernel and Cholesky
    K_final = compute_K(l_opt) + sigma_n_opt ** 2 * np.eye(n)
    
    try:
        L_final = np.linalg.cholesky(K_final)
        # Precompute alpha = K^-1 y for prediction
        alpha_final = np.linalg.solve(L_final, yn)
        alpha_final = np.linalg.solve(L_final.T, alpha_final)
    except np.linalg.LinAlgError:
        # Fallback to linear regression
        X_aug = np.column_stack([Xn, np.ones(n)])
        w, *_ = np.linalg.lstsq(X_aug, yn, rcond=None)
        def predict_fallback(X):
            Xn_test = normalize(X)
            X_aug_test = np.column_stack([Xn_test, np.ones(X.shape[0])])
            return (X_aug_test @ w) * y_std + y_mean
        return predict_fallback

    def predict(X):
        Xn_test = normalize(X)
        # Compute K(x_test, X_train)
        X_test_sq = np.sum(Xn_test ** 2, axis=1)
        sq_dists_test = X_test_sq[:, None] + Xn_sq[None, :] - 2.0 * (Xn_test @ Xn.T)
        sq_dists_test = np.maximum(sq_dists_test, 0.0)
        K_star = np.exp(-0.5 * sq_dists_test / (l_opt * l_opt))
        
        # Predict: mu = K_star @ K^-1 @ y
        # We have L_final such that K_final = L_final @ L_final.T
        # K^-1 @ y = alpha_final
        # But we need K_star @ alpha_final
        # Actually, for GP: f_new = K(X_new, X_train) @ K(X_train, X_train)^-1 @ y_train
        # alpha_final = K_final^-1 @ yn, so f_new = K_star @ alpha_final
        
        preds_normalized = K_star @ alpha_final
        preds = preds_normalized * y_std + y_mean
        return preds

    # Validate
    try:
        preds = predict(X_train)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            # Fallback to linear
            X_aug = np.column_stack([Xn, np.ones(n)])
            w, *_ = np.linalg.lstsq(X_aug, yn, rcond=None)
            def predict_fallback(X):
                Xn_test = normalize(X)
                X_aug_test = np.column_stack([Xn_test, np.ones(X.shape[0])])
                return (X_aug_test @ w) * y_std + y_mean
            return predict_fallback
    except Exception:
        X_aug = np.column_stack([Xn, np.ones(n)])
        w, *_ = np.linalg.lstsq(X_aug, yn, rcond=None)
        def predict_fallback(X):
            Xn_test = normalize(X)
            X_aug_test = np.column_stack([Xn_test, np.ones(X.shape[0])])
            return (X_aug_test @ w) * y_std + y_mean
        return predict_fallback

    return predict

# EVOLVE-BLOCK-END
