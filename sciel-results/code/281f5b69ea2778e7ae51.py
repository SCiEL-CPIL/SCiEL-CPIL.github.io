"""Search-based symbolic regression with time budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    # Safety margin: the evaluator kills at 330s. We are one of 10 calls.
    # Assuming ~30s per call is a safe average to stay within 300s total.
    # We use 28s to be safe.
    deadline = start_time + 28.0
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Helper to compute features for a given degree and extra terms
    def make_poly_features(X, degree=2):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Polynomial terms up to 'degree'
        # We generate monomials systematically to avoid duplicates and explosion
        # For d=1, deg=4: 1, x, x^2, x^3, x^4 (5 terms)
        # For d=3, deg=4: 1, x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2, ...
        
        # Simple approach: recursive or iterative generation of monomials
        # Given the small d (max 3) and small degree (max 4-5), we can just enumerate
        
        # To keep it simple and fast, we'll use a specific set of features
        # that covers common functions: polynomials, trig, exp, log, reciprocal
        
        # Let's build a richer basis if time permits, but start with polynomial
        if degree == 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
        elif degree == 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            for i in range(X.shape[1]):
                cols.append(X[:, i]**3)
                for j in range(i, X.shape[1]):
                    if j > i:
                        cols.append(X[:, i]**2 * X[:, j])
                        cols.append(X[:, i] * X[:, j]**2)
        elif degree == 4:
            # Degree 2 terms
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            # Degree 3 terms
            for i in range(X.shape[1]):
                cols.append(X[:, i]**3)
                for j in range(i, X.shape[1]):
                    if j > i:
                        cols.append(X[:, i]**2 * X[:, j])
                        cols.append(X[:, i] * X[:, j]**2)
            # Degree 4 terms
            for i in range(X.shape[1]):
                cols.append(X[:, i]**4)
                for j in range(i, X.shape[1]):
                    if j > i:
                        cols.append(X[:, i]**3 * X[:, j])
                        cols.append(X[:, i]**2 * X[:, j]**2)
                        cols.append(X[:, i] * X[:, j]**3)
        
        return np.column_stack(cols)
    
    # Baseline: degree-2 polynomial
    best_nrmse = np.inf
    best_predict_fn = None
    best_features_func = None
    best_w = None
    
    def try_model(features_func, X_tr, y_tr, use_ridge=False, lam=1e-6):
        nonlocal best_nrmse, best_predict_fn, best_features_func, best_w
        try:
            F = features_func(X_tr)
            if F.shape[1] > X_tr.shape[0]:
                # More features than samples, use ridge or SVD
                if use_ridge:
                    A = F.T @ F + lam * np.eye(F.shape[1])
                    b = F.T @ y_tr
                    w = np.linalg.solve(A, b)
                else:
                    w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            else:
                if use_ridge:
                    A = F.T @ F + lam * np.eye(F.shape[1])
                    b = F.T @ y_tr
                    w = np.linalg.solve(A, b)
                else:
                    w, *_ = np.linalg.lstsq(F, y_tr, rcond=None)
            
            pred = F @ w
            nrmse = np.mean((pred - y_tr)**2) / y_var
            
            if np.isfinite(nrmse) and nrmse < best_nrmse:
                best_nrmse = nrmse
                best_features_func = features_func
                best_w = w
                
                def make_pred_fn(feat_func, w):
                    return lambda X: feat_func(X) @ w
                
                best_predict_fn = make_pred_fn(features_func, w)
        except:
            pass

    # 1. Try Polynomial Degrees 2, 3, 4
    for deg in [2, 3, 4]:
        if time.time() > deadline - 5:
            break
        try_model(lambda X: make_poly_features(X, deg), X_train, y_train)
    
    # 2. Try Rich Basis: Polynomials + Trig + Exp/Log/Recip
    def make_rich_features(X):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            x = X[:, j]
            cols.append(x)
            cols.append(x**2)
            cols.append(x**3)
            cols.append(np.sin(x))
            cols.append(np.cos(x))
            cols.append(np.log(np.maximum(x, 1e-10)))
            cols.append(1.0 / np.maximum(x, 1e-10))
            # Maybe exp(x) if x is small, but x is in [1, 10] so exp can be huge.
            # We'll skip exp(x) to avoid numerical issues unless normalized.
            # Let's add normalized exp: exp((x - mean)/std) ? No, features must be deterministic on X.
            # Let's stick to the above.
        
        # Cross terms for pairs (linear product)
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.sin(X[:, i] * X[:, j]))
                cols.append(np.cos(X[:, i] * X[:, j]))
        
        return np.column_stack(cols)
    
    if time.time() < deadline - 5:
        try_model(make_rich_features, X_train, y_train, use_ridge=True)
    
    # 3. Try Kernel Ridge Regression (RBF)
    # This is powerful for non-polynomial shapes
    if time.time() < deadline - 10:
        try:
            from scipy.spatial.distance import cdist
            # Compute Gram matrix for training data
            dists = cdist(X_train, X_train, metric='sqeuclidean')
            
            # Try several bandwidths (gamma)
            # Heuristic: gamma = 1 / (d * var(X))
            # But we can just try a grid
            for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                if time.time() > deadline - 5:
                    break
                K = np.exp(-gamma * dists)
                lam = 1e-3
                K_reg = K + lam * np.eye(n)
                try:
                    alpha = np.linalg.solve(K_reg, y_train)
                    pred_k = K @ alpha
                    nrmse_k = np.mean((pred_k - y_train)**2) / y_var
                    
                    if np.isfinite(nrmse_k) and nrmse_k < best_nrmse:
                        best_nrmse = nrmse_k
                        
                        # Store data needed for prediction
                        X_tr_copy = X_train.copy()
                        alpha_copy = alpha.copy()
                        gamma_val = gamma
                        
                        def make_krbf_pred(X_test, X_tr, alpha, gamma):
                            dists_test = cdist(X_test, X_tr, metric='sqeuclidean')
                            K_test = np.exp(-gamma * dists_test)
                            return K_test @ alpha
                        
                        best_predict_fn = lambda X, X_tr=X_tr_copy, alpha=alpha_copy, gamma=gamma_val: make_krbf_pred(X, X_tr, alpha, gamma)
                except:
                    pass
        except:
            pass
    
    # Fallback if nothing worked (shouldn't happen)
    if best_predict_fn is None:
        # Return mean
        return lambda X: np.full(X.shape[0], y_mean)
    
    return best_predict_fn

# EVOLVE-BLOCK-END
