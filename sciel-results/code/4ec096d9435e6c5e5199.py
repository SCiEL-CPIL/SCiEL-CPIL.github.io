"""Rewritten program for Climate dynamics: unresolved-tendency closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fit_poly_ridge(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial to (x, y) using Ridge Regression.
    Returns coefficients in ascending powers of x.
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    # Center and scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
        
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    # Ridge Regression: (V.T V + alpha I) c = V.T y
    A = V.T @ V
    b = V.T @ y
    
    # Add regularization
    reg = alpha * np.eye(A.shape[0])
    # Ensure the intercept term (k=0) is not regularized if we want to preserve mean, 
    # but standard Ridge regularizes all. For polynomial fitting, regularizing all is fine.
    try:
        c_centered = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
        
    # Convert from centered/scaled basis to original basis
    # p(x) = sum_{k=0}^{degree} c_k * ((x-mu)/sigma)^k
    # Let z = (x-mu)/sigma => x = sigma*z + mu => z = (x - mu)/sigma
    # We need coefficients for x^0, x^1, ..., x^degree
    
    coeff_orig = np.zeros(degree + 1)
    a = 1.0 / sigma
    b_val = -mu / sigma
    
    from math import comb
    for k in range(degree + 1):
        if c_centered[k] == 0:
            continue
        # Expand (a*x + b_val)^k
        # Coefficients of (a*x + b_val)^k are binom(k, i) * a^i * b_val^(k-i) for x^i
        for i in range(k + 1):
            binom_k_i = comb(k, i)
            term_coeff = binom_k_i * (a ** i) * (b_val ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)
        
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        if len(x) >= 2:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except:
                coeff[0] = np.mean(y)
            return coeff
        elif len(x) >= 1:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)
            return coeff
        else:
            return np.zeros(6)

    n = len(x)
    
    # Use a single hold-out split for fast alpha selection
    # This is much faster than 5-fold CV and often sufficient
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    val_size = max(100, int(0.2 * n))
    if val_size >= n:
        val_size = n // 2
        
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    
    # Candidate alphas. Ridge alpha controls the strength of regularization.
    # We want to find a balance. Typical values for normalized data range from 1e-4 to 1e2.
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    best_alpha = 1e-3
    best_val_mse = np.inf
    best_coeff_train = np.zeros(6)
    
    degree = 5
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        try:
            # Fit on train
            c_train = _fit_poly_ridge(x_train, y_train, degree, alpha)
            
            # Evaluate on val
            y_pred_val = np.polynomial.polynomial.polyval(x_val, c_train)
            mse_val = np.mean((y_val - y_pred_val)**2)
            
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                best_alpha = alpha
                best_coeff_train = c_train
        except Exception:
            continue
            
    # Refit on all data with the best alpha
    try:
        best_coeff = _fit_poly_ridge(x, y, degree, best_alpha)
    except Exception:
        # Fallback to the train-only fit if full fit fails (unlikely)
        best_coeff = best_coeff_train
        
    # Pad to length 6 if necessary (though degree is 5, so length is 6)
    if len(best_coeff) < 6:
        padded = np.zeros(6)
        padded[:len(best_coeff)] = best_coeff
        best_coeff = padded
    elif len(best_coeff) > 6:
        best_coeff = best_coeff[:6]
        
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    return best_coeff[:6]
# EVOLVE-BLOCK-END
