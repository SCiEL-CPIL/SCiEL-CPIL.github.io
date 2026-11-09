"""Symbolic Regression using Gaussian Process Regression with Matern 5/2 Kernel."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _matern52_kernel(X1, X2, length_scales, variance):
    """
    Matern 5/2 kernel with ARD.
    X1: (n, d), X2: (m, d)
    length_scales: (d,) array
    variance: scalar
    """
    X1_exp = X1[:, np.newaxis, :]
    X2_exp = X2[np.newaxis, :, :]
    
    l_sq = length_scales[np.newaxis, np.newaxis, :] ** 2
    sq_dists = np.sum((X1_exp - X2_exp) ** 2 / l_sq, axis=2)
    
    r = np.sqrt(sq_dists)
    # Matern 5/2: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    return variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


def _fit_gpr(X_train, y_train, max_iter=50):
    """
    Fit Gaussian Process Regression.
    Optimizes hyperparameters: log_var, log_noise, log_ls.
    """
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_centered = (y_train - y_mean) / y_std
    
    # Normalize inputs to [0, 1]
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Initial hyperparameters
    # var ~ 1, noise ~ 1e-3, ls ~ 0.2
    log_var = 0.0
    log_noise = -5.0
    log_ls = np.full(d, np.log(0.2))
    
    def nlml(params):
        """Negative Log Marginal Likelihood."""
        log_var, log_noise, log_ls = params[0], params[1], params[2:]
        
        # Bounds check
        if log_var > 5 or log_var < -5: return 1e6
        if log_noise > 0 or log_noise < -10: return 1e6
        if np.any(log_ls > 2) or np.any(log_ls < -5): return 1e6
        
        var = np.exp(log_var)
        noise = np.exp(log_noise)
        ls = np.exp(log_ls)
        
        K = _matern52_kernel(X_norm, X_norm, ls, var) + noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
            
            # NLML = 0.5 * y^T K^-1 y + 0.5 * log(det(K)) + n/2 * log(2*pi)
            # log(det(K)) = 2 * sum(log(diag(L)))
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            quad_form = np.dot(y_centered, alpha)
            
            nlml_val = 0.5 * quad_form + 0.5 * log_det_K + 0.5 * n * np.log(2.0 * np.pi)
            
            if not np.isfinite(nlml_val):
                return 1e6
            return nlml_val
        except np.linalg.LinAlgError:
            return 1e6
    
    # Use Nelder-Mead for optimization
    initial_params = np.concatenate(([log_var, log_noise], log_ls))
    
    best_params = initial_params
    best_nlml = nlml(initial_params)
    
    # Run optimization
    try:
        res = minimize(nlml, initial_params, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_nlml:
            best_params = res.x
            best_nlml = res.fun
    except Exception:
        pass
    
    log_var, log_noise, log_ls = best_params[0], best_params[1], best_params[2:]
    var = np.exp(log_var)
    noise = np.exp(log_noise)
    ls = np.exp(log_ls)
    
    # Final fit
    K = _matern52_kernel(X_norm, X_norm, ls, var) + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails (should be rare with optimized noise)
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        
    def predict_gpr(X):
        X_min_ = X_train.min(axis=0, keepdims=True)
        X_max_ = X_train.max(axis=0, keepdims=True)
        X_range_ = X_max_ - X_min_
        X_range_[X_range_ < 1e-10] = 1.0
        X_norm_ = (X - X_min_) / X_range_
        
        K_star = _matern52_kernel(X_norm_, X_norm, ls, var)
        # Predictive mean: K_star K^-1 y
        preds = K_star @ alpha * y_std + y_mean
        return preds
    
    return predict_gpr


def _fit_ridge_poly(X_train, y_train):
    """
    Ridge Regression with Polynomial Features (degree 3).
    Used as a robust baseline.
    """
    n, d = X_train.shape
    
    # Generate polynomial features up to degree 3
    features = [np.ones(n)]
    for i in range(d):
        features.append(X_train[:, i])
    for total_deg in range(2, 4):
        for i in range(d):
            for j in range(i, d):
                if total_deg == 2:
                    if i == j:
                        features.append(X_train[:, i] ** 2)
                    else:
                        features.append(X_train[:, i] * X_train[:, j])
                elif total_deg == 3:
                    if i == j:
                        features.append(X_train[:, i] ** 3)
                    else:
                        # x_i^2 x_j and x_i x_j^2
                        features.append(X_train[:, i] ** 2 * X_train[:, j])
                        features.append(X_train[:, i] * X_train[:, j] ** 2)
                        
    F = np.column_stack(features)
    
    # Ridge regression
    alpha_ridge = 1e-2
    I = np.eye(F.shape[1])
    w = np.linalg.solve(F.T @ F + alpha_ridge * I, F.T @ y_train)
    
    def predict_poly(X):
        n_test = X.shape[0]
        feats = [np.ones(n_test)]
        for i in range(d):
            feats.append(X[:, i])
        for total_deg in range(2, 4):
            for i in range(d):
                for j in range(i, d):
                    if total_deg == 2:
                        if i == j:
                            feats.append(X[:, i] ** 2)
                        else:
                            feats.append(X[:, i] * X[:, j])
                    elif total_deg == 3:
                        if i == j:
                            feats.append(X[:, i] ** 3)
                        else:
                            feats.append(X[:, i] ** 2 * X[:, j])
                            feats.append(X[:, i] * X[:, j] ** 2)
        F_test = np.column_stack(feats)
        return F_test @ w
    
    return predict_poly


def fit(X_train, y_train):
    """
    Fit using GPR or Ridge Poly, selecting the best based on CV.
    """
    n, d = X_train.shape
    
    # Split for CV selection (simple 80/20 split)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    candidates = []
    
    # 1. GPR
    try:
        pred_gpr = _fit_gpr(X_tr, y_tr)
        val_pred_gpr = pred_gpr(X_val)
        mse_gpr = np.mean((val_pred_gpr - y_val) ** 2)
        candidates.append((mse_gpr, pred_gpr))
    except Exception:
        pass
        
    # 2. Ridge Poly
    try:
        pred_poly = _fit_ridge_poly(X_tr, y_tr)
        val_pred_poly = pred_poly(X_val)
        mse_poly = np.mean((val_pred_poly - y_val) ** 2)
        candidates.append((mse_poly, pred_poly))
    except Exception:
        pass
        
    if not candidates:
        # Fallback to mean
        y_mean = np.mean(y_train)
        def predict_mean(X):
            return np.full(len(X), y_mean)
        return predict_mean
        
    # Select best
    best_mse, best_pred = min(candidates, key=lambda x: x[0])
    
    return best_pred

# EVOLVE-BLOCK-END
