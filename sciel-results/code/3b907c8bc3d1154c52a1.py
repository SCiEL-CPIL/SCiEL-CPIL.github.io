import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin (330s limit)
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    n, d = X_train.shape
    
    # Handle constant y
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(np.asarray(X).shape[0], c)
        
    # Since the task specifies 1D inputs, we assume d=1.
    # If d > 1, we fall back to a simple linear model or ridge, but the contract says 1 input.
    if d != 1:
        # Fallback for unexpected dimensions: Ridge regression on raw features
        # This is unlikely to be hit given the contract, but ensures robustness.
        A = X_train.T @ X_train + 1e-6 * np.eye(d)
        b = X_train.T @ y_train
        w = np.linalg.solve(A, b)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X @ w
        return predict

    x = X_train[:, 0]
    y = y_train
    
    # Sort by x for interpolation methods
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicate x values by averaging y
    unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        y_sum = np.zeros(len(unique_x))
        np.add.at(y_sum, inverse, y_sorted)
        y_unique = y_sum / counts
        x_unique = unique_x
    else:
        x_unique = x_sorted
        y_unique = y_sorted
        
    n_unique = len(x_unique)
    
    # Define candidate models
    # Each model is a function that takes (x, y) and returns a predict function
    candidates = []
    
    # 1. Cubic Spline (Not-a-knot)
    def make_spline_predict(x_tr, y_tr):
        if len(x_tr) < 4:
            # Fallback to linear if too few points
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
        try:
            cs = CubicSpline(x_tr, y_tr, bc_type='not-a-knot')
            return lambda X: cs(X[:, 0] if X.ndim == 2 else X)
        except Exception:
            # Fallback to linear if spline fails
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
            
    candidates.append(("spline_nak", make_spline_predict))
    
    # 2. Cubic Spline (Natural)
    def make_spline_natural_predict(x_tr, y_tr):
        if len(x_tr) < 4:
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
        try:
            cs = CubicSpline(x_tr, y_tr, bc_type='natural')
            return lambda X: cs(X[:, 0] if X.ndim == 2 else X)
        except Exception:
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
            
    candidates.append(("spline_nat", make_spline_natural_predict))
    
    # 3. PCHIP
    def make_pchip_predict(x_tr, y_tr):
        if len(x_tr) < 4:
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
        try:
            pchip = PchipInterpolator(x_tr, y_tr)
            return lambda X: pchip(X[:, 0] if X.ndim == 2 else X)
        except Exception:
            return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
            
    candidates.append(("pchip", make_pchip_predict))
    
    # 4. Linear Interpolation
    def make_linear_predict(x_tr, y_tr):
        return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_tr, y_tr)
        
    candidates.append(("linear", make_linear_predict))
    
    # 5. Polynomial with Ridge (Degrees 2, 4, 6)
    def make_poly_predict(x_tr, y_tr, degree, alpha):
        V = np.vander(x_tr, N=degree+1, increasing=True)
        try:
            A = V.T @ V + alpha * np.eye(degree+1)
            b = V.T @ y_tr
            w = np.linalg.solve(A, b)
        except Exception:
            w = np.zeros(degree+1)
        
        def predict(X):
            x_new = X[:, 0] if X.ndim == 2 else X
            V_new = np.vander(x_new, N=degree+1, increasing=True)
            return V_new @ w
        return predict

    # Add a few polynomial candidates
    for deg in [2, 4, 6]:
        for alpha in [1e-4, 1e-2, 1.0]:
            candidates.append((f"poly_{deg}_{alpha}", lambda x_tr, y_tr, d=deg, a=alpha: make_poly_predict(x_tr, y_tr, d, a)))
            
    # 6. Gaussian Process (Single length scale)
    # GP is expensive (O(n^3)), but n=200 is small.
    # We will test a few length scales.
    x_range = np.max(x_unique) - np.min(x_unique)
    if x_range == 0:
        x_range = 1.0
        
    # Estimate typical scale
    # Use median distance between sorted points
    diffs = np.diff(x_unique)
    median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else x_range / 10.0
    
    gp_length_scales = [median_diff * 0.5, median_diff, median_diff * 2.0, x_range / 10.0, x_range / 5.0]
    # Deduplicate and filter
    gp_length_scales = list(set([ls for ls in gp_length_scales if ls > 1e-6]))
    
    def make_gp_predict(x_tr, y_tr, length_scale):
        y_mean = np.mean(y_tr)
        y_std = np.std(y_tr)
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (y_tr - y_mean) / y_std
        
        X_k = x_tr[:, np.newaxis]
        diff = X_k - X_k.T
        K = np.exp(-0.5 * (diff**2) / (length_scale**2))
        
        # Add small jitter for numerical stability
        K += 1e-8 * np.eye(len(x_tr))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
        except np.linalg.LinAlgError:
            # Fallback to least squares if Cholesky fails
            try:
                alpha = np.linalg.lstsq(K, y_scaled, rcond=None)[0]
            except Exception:
                return lambda X: np.full(len(X), y_mean)
        
        def predict(X):
            x_new = X[:, 0] if X.ndim == 2 else X
            diff_new = x_new[:, np.newaxis] - X_k
            K_star = np.exp(-0.5 * (diff_new**2) / (length_scale**2))
            y_pred_scaled = K_star @ alpha
            return y_pred_scaled * y_std + y_mean
            
        return predict

    for ls in gp_length_scales:
        candidates.append((f"gp_{ls:.4f}", lambda x_tr, y_tr, ls=ls: make_gp_predict(x_tr, y_tr, ls)))

    # Cross-Validation to select the best model
    # Use K-Fold CV
    n = len(x_unique)
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, k_folds)
    
    model_scores = {name: [] for name, _ in candidates}
    
    for i in range(k_folds):
        if time.time() > deadline:
            break
        val_idx = fold_indices[i]
        tr_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        
        x_val = x_unique[val_idx]
        y_val = y_unique[val_idx]
        x_tr = x_unique[tr_idx]
        y_tr = y_unique[tr_idx]
        
        for name, model_func in candidates:
            if time.time() > deadline:
                break
            try:
                predict_func = model_func(x_tr, y_tr)
                y_pred = predict_func(x_val.reshape(-1, 1))
                mse = np.mean((y_pred - y_val)**2)
                model_scores[name].append(mse)
            except Exception:
                model_scores[name].append(np.inf)
    
    # Select model with lowest mean CV MSE
    best_name = None
    best_score = np.inf
    for name, scores in model_scores.items():
        if len(scores) == 0:
            continue
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_name = name
    
    if best_name is None:
        # Fallback to simple linear interp
        return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_unique, y_unique)
        
    # Find the function for the best model
    best_model_func = None
    for name, func in candidates:
        if name == best_name:
            best_model_func = func
            break
            
    if best_model_func is None:
        return lambda X: np.interp(X[:, 0] if X.ndim == 2 else X, x_unique, y_unique)

    # Train the best model on ALL data
    final_predict = best_model_func(x_unique, y_unique)
    
    # Wrap to ensure correct output shape and finite values
    def final_wrapper(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            pred = final_predict(X)
            pred = np.asarray(pred).flatten()
            # Ensure finite
            pred = np.nan_to_num(pred, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
            return pred
        except Exception:
            return np.zeros(len(X))
            
    return final_wrapper
