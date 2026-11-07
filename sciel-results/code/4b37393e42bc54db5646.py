import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 295.0  # Safety margin from 330s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for kernel stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    
    model_scores = {}
    model_predictors = {}
    
    # ============ MODEL 1: Polynomial Kernel Ridge Regression ============
    def poly_kernel(X1, X2, degree, gamma, coef0):
        K = (gamma * (X1 @ X2.T) + coef0) ** degree
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def loo_cv_poly_kernel(X, y, degree, gamma, coef0, alpha):
        n = X.shape[0]
        K = poly_kernel(X, X, degree, gamma, coef0)
        A = K + alpha * np.eye(n)
        try:
            # Use efficient LOO formula for kernel ridge
            # h_ii = K_ii * (A^{-1})_ii
            # LOO prediction: y_i - h_ii * (y_i - f_i) where f_i is the full fit
            w = solve(A, y, assume_a='pos')
            f = K @ w
            diag = np.diag(K)
            # h_ii = diag * inv(A)_ii
            # inv(A)_ii can be computed as 1 / (A_ii + K_ii * (something))
            # More directly: h_ii = K_ii / (alpha + K_ii) for diagonal entries? No.
            # Use: h_ii = K_ii * (solve(A, e_i))_i
            # For efficiency, compute all h_ii at once
            # h_ii = diag(K) * diag(inv(A))
            # inv(A) = inv(K + alpha*I). By Sherman-Morrison or direct:
            # For small n, just compute
            inv_A = np.linalg.inv(A)
            h_ii = diag * np.diag(inv_A)
            y_loo = y - h_ii * (y - f)
            mse = np.mean((y_loo - y) ** 2)
            return mse
        except:
            return 1e10
    
    best_poly_score = np.inf
    best_poly_params = None
    
    if time.time() < deadline - 80:
        for degree in [2, 3, 4, 5]:
            for gamma in [0.5, 1.0, 2.0, 5.0, 10.0]:
                for coef0 in [0.5, 1.0, 2.0]:
                    for alpha in [0.01, 0.1, 1.0, 10.0]:
                        if time.time() > deadline - 60:
                            break
                        score = loo_cv_poly_kernel(X_tr_norm, y_tr, degree, gamma, coef0, alpha)
                        if score < best_poly_score:
                            best_poly_score = score
                            best_poly_params = (degree, gamma, coef0, alpha)
    
    if best_poly_params is not None:
        degree, gamma, coef0, alpha = best_poly_params
        K_full = poly_kernel(X_norm, X_norm, degree, gamma, coef0)
        A_full = K_full + alpha * np.eye(n)
        try:
            w_full = solve(A_full, y_centered, assume_a='pos')
            def predict_poly(X):
                X_test_norm = (X - min_vals) / ranges
                K_test = poly_kernel(X_test_norm, X_norm, degree, gamma, coef0)
                return K_test @ w_full + y_mean
            # Validate on holdout
            K_val_tr = poly_kernel(X_val_norm, X_tr_norm, degree, gamma, coef0)
            A_val = poly_kernel(X_tr_norm, X_tr_norm, degree, gamma, coef0) + alpha * np.eye(len(X_tr_norm))
            w_val = solve(A_val, y_tr, assume_a='pos')
            y_pred_val = K_val_tr @ w_val
            val_mse = np.mean((y_pred_val - y_val) ** 2)
            model_scores['poly'] = val_mse
            model_predictors['poly'] = predict_poly
        except:
            pass
    
    # ============ MODEL 2: Matern-5/2 GP ============
    def matern52_kernel(X1, X2, length_scale, signal_var):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        r = np.sqrt(sq_dists)
        K = signal_var * (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r / length_scale)
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def neg_log_likelihood_matern(params):
        length_scale = np.exp(params[0])
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        K = matern52_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_tr) * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    best_matern_score = np.inf
    best_matern_params = None
    
    if time.time() < deadline - 60:
        # Grid search
        for ls in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            for sv in [0.1, 0.5, 1.0, 5.0, 10.0]:
                for nv in [0.0001, 0.001, 0.01, 0.1]:
                    params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                    nll = neg_log_likelihood_matern(params)
                    if nll < best_matern_score:
                        best_matern_score = nll
                        best_matern_params = params
        
        # Refine with L-BFGS-B
        if best_matern_params is not None:
            try:
                result = minimize(neg_log_likelihood_matern, best_matern_params, method='L-BFGS-B',
                                 bounds=[(-5, 5), (-2, 5), (-9, -1)], maxiter=100)
                if result.fun < best_matern_score:
                    best_matern_score = result.fun
                    best_matern_params = result.x
            except:
                pass
    
    if best_matern_params is not None:
        length_scale = np.exp(best_matern_params[0])
        signal_var = np.exp(best_matern_params[1])
        noise_var = np.exp(best_matern_params[2])
        
        # Validate on holdout
        try:
            K_val_tr = matern52_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
            K_tr_tr = matern52_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
            w_val = solve(K_tr_tr, y_tr, assume_a='pos')
            y_pred_val = K_val_tr @ w_val
            val_mse = np.mean((y_pred_val - y_val) ** 2)
            
            # Refit on all data
            K_full = matern52_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
            w_full = solve(K_full, y_centered, assume_a='pos')
            
            def predict_matern(X):
                X_test_norm = (X - min_vals) / ranges
                K_test = matern52_kernel(X_test_norm, X_norm, length_scale, signal_var)
                return K_test @ w_full + y_mean
            
            model_scores['matern'] = val_mse
            model_predictors['matern'] = predict_matern
        except:
            pass
    
    # ============ MODEL 3: RBF GP ============
    def rbf_kernel(X1, X2, length_scale, signal_var):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K = signal_var * np.exp(-sq_dists / (2 * length_scale**2))
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def neg_log_likelihood_rbf(params):
        length_scale = np.exp(params[0])
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        K = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_tr) * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    best_rbf_score = np.inf
    best_rbf_params = None
    
    if time.time() < deadline - 40:
        for ls in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            for sv in [0.1, 0.5, 1.0, 5.0, 10.0]:
                for nv in [0.0001, 0.001, 0.01, 0.1]:
                    params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                    nll = neg_log_likelihood_rbf(params)
                    if nll < best_rbf_score:
                        best_rbf_score = nll
                        best_rbf_params = params
        
        if best_rbf_params is not None:
            try:
                result = minimize(neg_log_likelihood_rbf, best_rbf_params, method='L-BFGS-B',
                                 bounds=[(-5, 5), (-2, 5), (-9, -1)], maxiter=100)
                if result.fun < best_rbf_score:
                    best_rbf_score = result.fun
                    best_rbf_params = result.x
            except:
                pass
    
    if best_rbf_params is not None:
        length_scale = np.exp(best_rbf_params[0])
        signal_var = np.exp(best_rbf_params[1])
        noise_var = np.exp(best_rbf_params[2])
        
        try:
            K_val_tr = rbf_kernel(X_val_norm, X_tr_norm, length_scale, signal_var)
            K_tr_tr = rbf_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
            w_val = solve(K_tr_tr, y_tr, assume_a='pos')
            y_pred_val = K_val_tr @ w_val
            val_mse = np.mean((y_pred_val - y_val) ** 2)
            
            K_full = rbf_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(n)
            w_full = solve(K_full, y_centered, assume_a='pos')
            
            def predict_rbf(X):
                X_test_norm = (X - min_vals) / ranges
                K_test = rbf_kernel(X_test_norm, X_norm, length_scale, signal_var)
                return K_test @ w_full + y_mean
            
            model_scores['rbf'] = val_mse
            model_predictors['rbf'] = predict_rbf
        except:
            pass
    
    # ============ MODEL 4: Feature-based linear regression (polynomial features) ============
    def generate_features(X, degree):
        """Generate polynomial features up to given degree."""
        n, d = X.shape
        features = [np.ones(n)]
        
        for deg in range(1, degree + 1):
            for combo in _combinations_with_replacement(range(d), deg):
                term = np.ones(n)
                for idx in combo:
                    term *= X[:, idx]
                features.append(term)
        
        return np.column_stack(features)
    
    def _combinations_with_replacement(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = range(indices[i] + 1, indices[i] + 1 + r - i)
            yield tuple(pool[i] for i in indices)
    
    best_feat_score = np.inf
    best_feat_predict = None
    best_feat_degree = None
    
    if time.time() < deadline - 30:
        for degree in [2, 3, 4]:
            try:
                X_tr_feat = generate_features(X_tr_norm, degree)
                X_val_feat = generate_features(X_val_norm, degree)
                
                # Ridge regression
                alpha = 0.1
                A = X_tr_feat.T @ X_tr_feat + alpha * np.eye(X_tr_feat.shape[1])
                b = X_tr_feat.T @ y_tr
                w = solve(A, b, assume_a='pos')
                
                y_pred_val = X_val_feat @ w
                val_mse = np.mean((y_pred_val - y_val) ** 2)
                
                if val_mse < best_feat_score:
                    # Refit on all data
                    X_full_feat = generate_features(X_norm, degree)
                    A_full = X_full_feat.T @ X_full_feat + alpha * np.eye(X_full_feat.shape[1])
                    b_full = X_full_feat.T @ y_centered
                    w_full = solve(A_full, b_full, assume_a='pos')
                    
                    best_feat_score = val_mse
                    best_feat_degree = degree
                    best_feat_predict = (w_full, degree)
            except:
                continue
    
    if best_feat_predict is not None:
        w_full, degree = best_feat_predict
        def predict_feat(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm, degree)
            return X_test_feat @ w_full + y_mean
        model_scores['feat'] = best_feat_score
        model_predictors['feat'] = predict_feat
    
    # ============ Select best model ============
    if not model_scores:
        # Fallback: simple linear regression
        X_design = np.hstack([np.ones((n, 1)), X_norm])
        w = np.linalg.lstsq(X_design, y_centered, rcond=None)[0]
        def predict_fallback(X):
            X_test_norm = (X - min_vals) / ranges
            X_design_test = np.hstack([np.ones((X.shape[0], 1)), X_test_norm])
            return X_design_test @ w + y_mean
        return predict_fallback
    
    # Select model with lowest validation score
    best_model_name = min(model_scores, key=model_scores.get)
    return model_predictors[best_model_name]
