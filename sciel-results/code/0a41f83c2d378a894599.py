import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    d = X.shape[1]
    n = len(y)
    
    # Handle potential multi-dimensional input (though task says 1D)
    if d > 1:
        # Fallback to KRR for multi-dim
        return _fit_krr_md(X, y)

    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values
    # If duplicates exist, average y
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < n:
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        x_sorted = unique_x
        y_sorted = y_avg
        n = len(x_sorted)
    
    # Time check
    if time.time() > deadline - 10:
        # Emergency fallback: simple interpolation
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return np.interp(X_pred[:, 0], x_sorted, y_sorted)
        return predict

    # Split for validation
    # Use a deterministic split to ensure reproducibility
    n_val = max(10, int(0.1 * n))
    n_tr = n - n_val
    
    # Ensure we don't use the same points for train and val
    # Simple random shuffle with fixed seed
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]
    
    x_tr = x_sorted[tr_idx]
    y_tr = y_sorted[tr_idx]
    x_val = x_sorted[val_idx]
    y_val = y_sorted[val_idx]
    
    # Sort training data again for splines
    tr_sort = np.argsort(x_tr)
    x_tr_s = x_tr[tr_sort]
    y_tr_s = y_tr[tr_sort]
    
    best_val_mse = np.inf
    best_model_type = 'spline'
    best_params = {}
    
    # 1. Cubic Spline (not-a-knot)
    try:
        cs = CubicSpline(x_tr_s, y_tr_s, bc_type='not-a-knot')
        y_pred = cs(x_val)
        mse = np.mean((y_pred - y_val)**2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_model_type = 'spline_nk'
            best_params = {}
    except Exception:
        pass
    
    # 2. PCHIP (Monotone Cubic)
    try:
        pchip = PchipInterpolator(x_tr_s, y_tr_s)
        y_pred = pchip(x_val)
        mse = np.mean((y_pred - y_val)**2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_model_type = 'pchip'
            best_params = {}
    except Exception:
        pass
        
    # 3. Cubic Spline (natural) - just in case
    try:
        cs_nat = CubicSpline(x_tr_s, y_tr_s, bc_type='natural')
        y_pred = cs_nat(x_val)
        mse = np.mean((y_pred - y_val)**2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_model_type = 'spline_nat'
            best_params = {}
    except Exception:
        pass

    # 4. Polynomial Fitting
    # Try degrees 2 to 12
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 20:
            break
        try:
            # Use np.polyfit which is least squares
            coeffs = np.polyfit(x_tr_s, y_tr_s, deg)
            # Evaluate
            y_pred = np.polyval(coeffs, x_val)
            mse = np.mean((y_pred - y_val)**2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'poly'
                best_params = {'deg': deg, 'coeffs': coeffs}
        except Exception:
            pass

    # 5. Kernel Ridge Regression (Gaussian)
    # Heuristic for gamma: 1 / (2 * sigma^2). 
    # Let's try a range of gammas.
    # The "scale" of the data is roughly (max-min)
    x_range = x_sorted.max() - x_sorted.min()
    if x_range == 0:
        x_range = 1.0
    
    # Gammas corresponding to different effective widths
    # width = sqrt(1/(2*gamma))
    # If width is small, it's more local (interpolative). If large, it's smoother.
    # For 200 points in range R, average spacing is R/200.
    # A good local width is around spacing.
    avg_spacing = x_range / max(n, 1)
    
    # Try gammas that correspond to widths: 0.1*spacing, 0.5*spacing, 1*spacing, 2*spacing, 5*spacing, 10*spacing
    widths = [0.1 * avg_spacing, 0.5 * avg_spacing, 1.0 * avg_spacing, 
              2.0 * avg_spacing, 5.0 * avg_spacing, 10.0 * avg_spacing, 
              0.01 * avg_spacing]
    
    for width in widths:
        if time.time() > deadline - 30:
            break
        gamma = 1.0 / (2.0 * width**2)
        try:
            # Build Kernel Matrix K
            # K[i,j] = exp(-gamma * (x_i - x_j)^2)
            diff = x_tr_s[:, None] - x_tr_s[None, :]
            K = np.exp(-gamma * diff**2)
            
            # Regularization
            lam = 1e-6  # Small regularization for stability
            A = K + lam * np.eye(len(x_tr_s))
            
            # Solve
            w = solve(A, y_tr_s)
            
            # Predict
            diff_val = x_val[:, None] - x_tr_s[None, :]
            K_val = np.exp(-gamma * diff_val**2)
            y_pred = K_val @ w
            
            mse = np.mean((y_pred - y_val)**2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'krr'
                best_params = {'gamma': gamma, 'width': width, 'lam': lam}
        except Exception:
            pass

    # Refit on ALL data with the best model
    if best_model_type == 'spline_nk':
        final_cs = CubicSpline(x_sorted, y_sorted, bc_type='not-a-knot')
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return final_cs(X_pred[:, 0])
            
    elif best_model_type == 'pchip':
        final_pchip = PchipInterpolator(x_sorted, y_sorted)
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return final_pchip(X_pred[:, 0])
            
    elif best_model_type == 'spline_nat':
        final_cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return final_cs(X_pred[:, 0])
            
    elif best_model_type == 'poly':
        deg = best_params['deg']
        final_coeffs = np.polyfit(x_sorted, y_sorted, deg)
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return np.polyval(final_coeffs, X_pred[:, 0])
            
    elif best_model_type == 'krr':
        gamma = best_params['gamma']
        lam = best_params['lam']
        diff_all = x_sorted[:, None] - x_sorted[None, :]
        K_all = np.exp(-gamma * diff_all**2)
        A_all = K_all + lam * np.eye(n)
        try:
            w_final = solve(A_all, y_sorted)
        except:
            # Fallback to interpolation if solve fails
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return np.interp(X_pred[:, 0], x_sorted, y_sorted)
            return predict
            
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            x_pred = X_pred[:, 0]
            diff_pred = x_pred[:, None] - x_sorted[None, :]
            K_pred = np.exp(-gamma * diff_pred**2)
            return K_pred @ w_final
            
    else:
        # Fallback: Linear Interpolation
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            return np.interp(X_pred[:, 0], x_sorted, y_sorted)

    return predict

def _fit_krr_md(X, y):
    """Fallback for multi-dimensional inputs."""
    n = len(y)
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(10, int(0.1 * n))
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]
    
    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    
    best_mse = np.inf
    best_gamma = 1.0
    
    # Try a few gammas
    # Estimate scale
    dists = np.linalg.norm(X_tr[:, None, :] - X_tr[None, :, :], axis=2)
    median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    
    gammas = [1/(2*median_dist**2) * f for f in [0.1, 0.5, 1, 2, 5, 10]]
    
    for gamma in gammas:
        try:
            # RBF Kernel
            sq_dists_tr = np.sum(X_tr**2, axis=1)[:, None] + np.sum(X_tr**2, axis=1)[None, :] - 2 * (X_tr @ X_tr.T)
            np.maximum(sq_dists_tr, 0, out=sq_dists_tr)
            K_tr = np.exp(-gamma * sq_dists_tr)
            
            lam = 1e-4
            A = K_tr + lam * np.eye(len(X_tr))
            w = solve(A, y_tr)
            
            sq_dists_val = np.sum(X_val**2, axis=1)[:, None] + np.sum(X_tr**2, axis=1)[None, :] - 2 * (X_val @ X_tr.T)
            np.maximum(sq_dists_val, 0, out=sq_dists_val)
            K_val = np.exp(-gamma * sq_dists_val)
            
            y_pred = K_val @ w
            mse = np.mean((y_pred - y_val)**2)
            
            if mse < best_mse:
                best_mse = mse
                best_gamma = gamma
        except:
            continue
            
    # Refit on all data
    gamma = best_gamma
    sq_dists_all = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X @ X.T)
    np.maximum(sq_dists_all, 0, out=sq_dists_all)
    K_all = np.exp(-gamma * sq_dists_all)
    lam = 1e-4
    A_all = K_all + lam * np.eye(n)
    try:
        w_final = solve(A_all, y)
    except:
        mean_y = np.mean(y)
        def predict(X_pred):
            return np.full(len(X_pred), mean_y)
        return predict
        
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)
        
        sq_dists_pred = np.sum(X_pred**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * (X_pred @ X.T)
        np.maximum(sq_dists_pred, 0, out=sq_dists_pred)
        K_pred = np.exp(-gamma * sq_dists_pred)
        
        return K_pred @ w_final
        
    return predict
