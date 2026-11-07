import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin from 330s limit
    
    n_samples, n_features = X_train.shape
    
    # Helper: generate polynomial features up to given degree
    def get_poly_features(X, degree=5):
        cols = [np.ones(len(X))]
        d = X.shape[1]
        # Generate all monomials up to given degree
        for deg in range(1, degree + 1):
            for indices in itertools.combinations_with_replacement(range(d), deg):
                term = np.ones(len(X))
                for idx in indices:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    # Pre-compute squared norms for distance calculations
    X_sq_norms = np.sum(X_train ** 2, axis=1)
    
    # Pre-compute all pairwise squared distances
    dist_sq = X_sq_norms[:, None] + X_sq_norms[None, :] - 2.0 * (X_train @ X_train.T)
    np.maximum(dist_sq, 0, out=dist_sq)
    
    # Model 1: Polynomial Ridge Regression with various degrees
    def fit_poly_ridge(X, y, degree, lam):
        P = get_poly_features(X, degree)
        n_params = P.shape[1]
        A = P.T @ P + lam * np.eye(n_params)
        b = P.T @ y
        try:
            w = solve(A, b)
            return w, P
        except Exception:
            return None, P
    
    # Model 2: RBF Kernel Ridge Regression
    def fit_rbf_ridge(X, y, sigma, lam):
        X_sq = np.sum(X ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d_sq, 0, out=d_sq)
        K = np.exp(-d_sq / (2.0 * sigma ** 2))
        n_params = K.shape[1]
        A = K + lam * np.eye(n_params)
        try:
            w = solve(A, y)
            return w, K
        except Exception:
            return None, K
    
    # Model 3: Matern-5/2 Gaussian Process (using kernel ridge as surrogate)
    def fit_matern_ridge(X, y, length_scale, variance, noise):
        X_sq = np.sum(X ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d_sq, 0, out=d_sq)
        d = np.sqrt(d_sq) / length_scale
        K = variance * (1 + d + d**2 / 3.0) * np.exp(-d)
        K += noise * np.eye(n_samples)
        try:
            w = solve(K, y)
            return w, K
        except Exception:
            return None, K
    
    # Model 4: Rational Function Approximator (ratio of two linear basis expansions)
    def fit_rational(X, y, degree_num, degree_den):
        # N(x) / D(x) where N and D are polynomials
        P_num = get_poly_features(X, degree_num)
        P_den = get_poly_features(X, degree_den)
        # Add bias term to denominator
        P_den = np.hstack([P_den, np.ones((len(X), 1))])
        
        # We want to minimize ||y * D(x) - N(x)||^2
        # This is not linear, so use iterative approach or grid search
        # For simplicity, use a grid search over some parameters
        
        # Actually, let's use a different approach: fit N(x) = y * D(x) using linear regression
        # But D(x) is unknown. Let's use a simple heuristic:
        # Fit D(x) first to be close to 1, then fit N(x) = y * D(x)
        
        # Simpler: just use polynomial regression with high degree as a proxy
        # Or use a fixed denominator like 1 + sum(x_i^2)
        
        # For now, skip rational and use high-degree polynomial instead
        return None, None
    
    # Cross-validation helper
    def cross_validate(model_func, param_grid, X, y, n_folds=5):
        rng = np.random.RandomState(42)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        fold_indices = np.array_split(indices, n_folds)
        
        best_score = np.inf
        best_params = None
        
        for params in param_grid:
            if time.time() > deadline - 10.0:
                break
            
            cv_errors = []
            for i in range(n_folds):
                val_idx = fold_indices[i]
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                
                X_tr = X[train_idx]
                y_tr = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                result = model_func(X_tr, y_tr, *params)
                if result is None:
                    cv_errors.append(np.inf)
                    continue
                
                w, K_tr = result
                
                # Predict on validation set
                if isinstance(K_tr, np.ndarray) and K_tr.shape[0] == len(X_tr):
                    # For RBF/Matern: compute kernel between val and train
                    X_tr_sq = np.sum(X_tr ** 2, axis=1)
                    X_val_sq = np.sum(X_val ** 2, axis=1)
                    d_val_sq = X_val_sq[:, None] + X_tr_sq[None, :] - 2.0 * (X_val @ X_tr.T)
                    np.maximum(d_val_sq, 0, out=d_val_sq)
                    
                    if 'rbf' in str(params):
                        sigma = params[0]
                        K_val = np.exp(-d_val_sq / (2.0 * sigma ** 2))
                    elif 'matern' in str(params):
                        ls, var, noise = params
                        d_val = np.sqrt(d_val_sq) / ls
                        K_val = var * (1 + d_val + d_val**2 / 3.0) * np.exp(-d_val)
                    else:
                        cv_errors.append(np.inf)
                        continue
                    
                    y_pred = K_val @ w
                else:
                    # For polynomial: generate features for val
                    P_val = get_poly_features(X_val, params[0])
                    y_pred = P_val @ w
                
                mse = np.mean((y_pred - y_val) ** 2)
                cv_errors.append(mse)
            
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_score:
                best_score = mean_cv
                best_params = params
        
        return best_params, best_score
    
    # Define models and their parameter grids
    models = []
    
    # Model 1: Polynomial Ridge
    def poly_ridge_func(X, y, degree, lam):
        P = get_poly_features(X, degree)
        n_params = P.shape[1]
        A = P.T @ P + lam * np.eye(n_params)
        b = P.T @ y
        try:
            w = solve(A, b)
            return w, ('poly', degree)
        except Exception:
            return None, None
    
    poly_params = []
    for degree in [2, 3, 4, 5]:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            poly_params.append((degree, lam))
    
    # Model 2: RBF Ridge
    def rbf_ridge_func(X, y, sigma, lam):
        X_sq = np.sum(X ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d_sq, 0, out=d_sq)
        K = np.exp(-d_sq / (2.0 * sigma ** 2))
        n_params = K.shape[1]
        A = K + lam * np.eye(n_params)
        try:
            w = solve(A, y)
            return w, ('rbf', sigma)
        except Exception:
            return None, None
    
    rbf_params = []
    for sigma in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
            rbf_params.append((sigma, lam))
    
    # Model 3: Matern-5/2 Ridge
    def matern_ridge_func(X, y, length_scale, variance, noise):
        X_sq = np.sum(X ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d_sq, 0, out=d_sq)
        d = np.sqrt(d_sq) / length_scale
        K = variance * (1 + d + d**2 / 3.0) * np.exp(-d)
        K += noise * np.eye(len(X))
        try:
            w = solve(K, y)
            return w, ('matern', length_scale, variance, noise)
        except Exception:
            return None, None
    
    matern_params = []
    for ls in [0.5, 1.0, 2.0, 5.0]:
        for var in [1.0, 5.0, 10.0]:
            for noise in [1e-6, 1e-4, 1e-3]:
                matern_params.append((ls, var, noise))
    
    # Evaluate each model type
    best_model_type = None
    best_model_params = None
    best_cv_score = np.inf
    
    # Try polynomial
    if time.time() < deadline - 20.0:
        best_poly_params, poly_score = cross_validate(poly_ridge_func, poly_params, X_train, y_train)
        if poly_score < best_cv_score:
            best_cv_score = poly_score
            best_model_type = 'poly'
            best_model_params = best_poly_params
    
    # Try RBF
    if time.time() < deadline - 20.0:
        best_rbf_params, rbf_score = cross_validate(rbf_ridge_func, rbf_params, X_train, y_train)
        if rbf_score < best_cv_score:
            best_cv_score = rbf_score
            best_model_type = 'rbf'
            best_model_params = best_rbf_params
    
    # Try Matern
    if time.time() < deadline - 20.0:
        best_matern_params, matern_score = cross_validate(matern_ridge_func, matern_params, X_train, y_train)
        if matern_score < best_cv_score:
            best_cv_score = matern_score
            best_model_type = 'matern'
            best_model_params = best_matern_params
    
    # Final fit on all data with best model
    if best_model_type is None:
        # Fallback: simple mean
        mean_y = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(len(X), mean_y)
        return predict
    
    # Fit final model
    if best_model_type == 'poly':
        degree, lam = best_model_params
        P = get_poly_features(X_train, degree)
        n_params = P.shape[1]
        A = P.T @ P + lam * np.eye(n_params)
        b = P.T @ y_train
        try:
            w_final = solve(A, b)
        except Exception:
            w_final = np.linalg.lstsq(A, b, rcond=None)[0]
        
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            P_pred = get_poly_features(X, degree)
            return P_pred @ w_final
    
    elif best_model_type == 'rbf':
        sigma, lam = best_model_params
        X_sq = np.sum(X_train ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X_train @ X_train.T)
        np.maximum(d_sq, 0, out=d_sq)
        K = np.exp(-d_sq / (2.0 * sigma ** 2))
        n_params = K.shape[1]
        A = K + lam * np.eye(n_params)
        try:
            w_final = solve(A, y_train)
        except Exception:
            w_final = np.linalg.lstsq(A, y_train, rcond=None)[0]
        
        train_X = X_train
        final_sigma = sigma
        
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_sq_pred = np.sum(X ** 2, axis=1)
            train_sq = np.sum(train_X ** 2, axis=1)
            d_sq_pred = X_sq_pred[:, None] + train_sq[None, :] - 2.0 * (X @ train_X.T)
            np.maximum(d_sq_pred, 0, out=d_sq_pred)
            K_pred = np.exp(-d_sq_pred / (2.0 * final_sigma ** 2))
            return K_pred @ w_final
    
    elif best_model_type == 'matern':
        ls, var, noise = best_model_params
        X_sq = np.sum(X_train ** 2, axis=1)
        d_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X_train @ X_train.T)
        np.maximum(d_sq, 0, out=d_sq)
        d = np.sqrt(d_sq) / ls
        K = var * (1 + d + d**2 / 3.0) * np.exp(-d)
        K += noise * np.eye(n_samples)
        try:
            w_final = solve(K, y_train)
        except Exception:
            w_final = np.linalg.lstsq(K, y_train, rcond=None)[0]
        
        train_X = X_train
        final_ls = ls
        final_var = var
        
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_sq_pred = np.sum(X ** 2, axis=1)
            train_sq = np.sum(train_X ** 2, axis=1)
            d_sq_pred = X_sq_pred[:, None] + train_sq[None, :] - 2.0 * (X @ train_X.T)
            np.maximum(d_sq_pred, 0, out=d_sq_pred)
            d_pred = np.sqrt(d_sq_pred) / final_ls
            K_pred = final_var * (1 + d_pred + d_pred**2 / 3.0) * np.exp(-d_pred)
            return K_pred @ w_final
    
    else:
        mean_y = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(len(X), mean_y)
    
    return predict
