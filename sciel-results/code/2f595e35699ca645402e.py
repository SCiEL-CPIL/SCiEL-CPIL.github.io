import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # If n is too small for spline, fall back to direct fit
    if n < 50:
        # Direct ridge regression
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12: x_std = 1.0
        x_norm = (x - x_mean) / x_std
        V = np.vander(x_norm, N=6, increasing=True)
        # Simple ridge
        lam = 1e-4
        A = V.T @ V + lam * np.eye(6)
        b = V.T @ y
        try:
            coeff_norm = np.linalg.solve(A, b)
        except:
            coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Convert back
        coeff_orig = np.zeros(6, dtype=np.float64)
        for i in range(6):
            ci = coeff_norm[i]
            if abs(ci) < 1e-15: continue
            ci_scaled = ci / (x_std ** i)
            for j in range(i + 1):
                binom_val = 1.0
                for k in range(j):
                    binom_val = binom_val * (i - k) / (k + 1)
                term = binom_val * ((-x_mean) ** (i - j)) * ci_scaled
                coeff_orig[j] += term
        return np.nan_to_num(coeff_orig)

    # Strategy: Smooth y with a spline, then fit polynomial
    # Sort by x for spline
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Try a few smoothing factors
    # s is the sum of squared residuals allowed
    # Heuristic: s = alpha * n * variance(y)
    var_y = np.var(y)
    candidates_s = [0.01 * n * var_y, 0.1 * n * var_y, 1.0 * n * var_y]
    
    best_poly = None
    best_val_mse = np.inf
    
    # Use a held-out set for validation of the smoothing choice
    # Since we are smoothing, we can just fit on all data with a regularizer,
    # but let's stick to the "denoise then fit" idea.
    
    # We will fit the polynomial on the smoothed data.
    # To choose the best smoothing, we can use cross-validation on the polynomial fit
    # or just pick a moderate smoothing.
    
    # Let's try a simple approach:
    # 1. Fit spline with moderate s
    # 2. Get y_smooth
    # 3. Fit poly to y_smooth
    
    # We need to be careful not to over-smooth.
    # Let's try 3 different s values and evaluate the resulting polynomial on a validation split of the ORIGINAL data?
    # No, the polynomial is supposed to approximate the true tendency.
    # If we smooth, we reduce variance but add bias.
    # The goal is to minimize MSE on validation.
    
    # Let's use a validation split of the original data to select the best (s, lambda) combination.
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = max(100, min(n // 10, 5000))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train_orig = x[train_idx]
    y_train_orig = y[train_idx]
    x_val_orig = x[val_idx]
    y_val_orig = y[val_idx]
    
    # Standardize for polynomial fit stability
    x_mean = np.mean(x_train_orig)
    x_std = np.std(x_train_orig)
    if x_std < 1e-12: x_std = 1.0
    
    # We will fit the spline on the training part, then evaluate polynomial on validation part.
    # This is proper CV.
    
    # Sort training data for spline
    sort_idx_train = np.argsort(x_train_orig)
    x_train_sorted = x_train_orig[sort_idx_train]
    y_train_sorted = y_train_orig[sort_idx_train]
    
    var_y_train = np.var(y_train_sorted)
    
    # Smoothing candidates
    s_candidates = [0.01 * len(x_train_sorted) * var_y_train, 
                    0.1 * len(x_train_sorted) * var_y_train,
                    1.0 * len(x_train_sorted) * var_y_train]
    
    # Regularization candidates for the final polynomial fit
    # Since we are fitting to smoothed data, we can use very small regularization
    lam_candidates = [1e-6, 1e-4, 1e-2]
    
    best_score = np.inf
    best_s = s_candidates[1]
    best_lam = lam_candidates[1]
    
    for s in s_candidates:
        if time.time() - t_start > deadline - 10:
            break
        try:
            # Fit spline
            # k=3 is standard
            # Note: UnivariateSpline requires x to be increasing
            spline = UnivariateSpline(x_train_sorted, y_train_sorted, s=s, k=3)
            
            # Get smoothed y for training data
            y_train_smooth = spline(x_train_orig)
            
            for lam in lam_candidates:
                if time.time() - t_start > deadline - 5:
                    break
                
                # Fit polynomial to smoothed data
                x_train_norm = (x_train_orig - x_mean) / x_std
                V_train = np.vander(x_train_norm, N=6, increasing=True)
                
                A = V_train.T @ V_train + lam * np.eye(6)
                b = V_train.T @ y_train_smooth
                
                try:
                    coeff_norm = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                if not np.all(np.isfinite(coeff_norm)):
                    continue
                
                # Evaluate on validation set
                # We need to evaluate the POLYNOMIAL on the validation set
                # The polynomial is fitted to the smoothed training data.
                # The validation set has original noisy y.
                x_val_norm = (x_val_orig - x_mean) / x_std
                V_val = np.vander(x_val_norm, N=6, increasing=True)
                pred_val = V_val @ coeff_norm
                val_mse = np.mean((pred_val - y_val_orig) ** 2)
                
                if val_mse < best_score:
                    best_score = val_mse
                    best_s = s
                    best_lam = lam
                    
        except Exception:
            continue

    # Refit on ALL data with best parameters
    if time.time() - t_start > deadline - 5:
        # Fallback to simple ridge
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12: x_std = 1.0
        x_norm = (x - x_mean) / x_std
        V = np.vander(x_norm, N=6, increasing=True)
        A = V.T @ V + best_lam * np.eye(6)
        b = V.T @ y
        try:
            coeff_norm = np.linalg.solve(A, b)
        except:
            coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        sort_idx_all = np.argsort(x)
        x_all_sorted = x[sort_idx_all]
        y_all_sorted = y[sort_idx_all]
        
        try:
            spline = UnivariateSpline(x_all_sorted, y_all_sorted, s=best_s, k=3)
            y_all_smooth = spline(x)
        except Exception:
            y_all_smooth = y # Fallback if spline fails
            
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12: x_std = 1.0
        x_norm = (x - x_mean) / x_std
        V = np.vander(x_norm, N=6, increasing=True)
        
        A = V.T @ V + best_lam * np.eye(6)
        b = V.T @ y_all_smooth
        
        try:
            coeff_norm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]

    if not np.all(np.isfinite(coeff_norm)):
        coeff_norm = np.zeros(6)
        
    # Convert from normalized x to original x scale
    coeff_orig = np.zeros(6, dtype=np.float64)
    for i in range(6):
        ci = coeff_norm[i]
        if abs(ci) < 1e-15:
            continue
        ci_scaled = ci / (x_std ** i)
        for j in range(i + 1):
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
            term = binom_val * ((-x_mean) ** (i - j)) * ci_scaled
            coeff_orig[j] += term
            
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
        
    return coeff_orig
