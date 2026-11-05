import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import stats

DEADLINE = 145.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _check_deadline():
    return _elapsed() > DEADLINE

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using robust scaling and cross-validated Ridge regression in Chebyshev basis.
    Returns 6 coefficients in ASCENDING powers of x.
    """
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any non-finite values
    mask = np.isfinite(training_x) & np.isfinite(training_subgrid_tendency)
    training_x = training_x[mask]
    training_subgrid_tendency = training_subgrid_tendency[mask]
    
    n = training_x.size
    if n == 0:
        return np.zeros(6)
    
    if n < 6:
        deg = max(1, min(n - 1, 5))
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg)
            coeff[:deg+1] = c
        except Exception:
            coeff[0] = np.mean(training_subgrid_tendency)
        return coeff

    # 1. Robust Scaling to [-1, 1]
    # Use percentiles to avoid outlier sensitivity
    p_low = np.percentile(training_x, 1)
    p_high = np.percentile(training_x, 99)
    x_range = p_high - p_low
    
    if x_range < 1e-12:
        return np.array([np.mean(training_subgrid_tendency)] + [0.0]*5)
        
    # Map x to z in [-1, 1] based on percentiles
    # z = 2*(x - p_low)/(p_high - p_low) - 1
    z = 2.0 * (training_x - p_low) / x_range - 1.0
    
    # 2. Cross-Validated Ridge Regression in Chebyshev Basis
    # We want to find the best alpha (regularization strength)
    # Use 5-Fold CV
    n_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    # Handle remainder
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[:n - fold_size*n_folds]])
        
    # Range of alphas to test
    # Log spaced from very small (LSQ) to large (strong regularization)
    alphas = np.logspace(-10, 5, 50)
    
    best_alpha = 0.0
    best_cv_mse = np.inf
    
    # Pre-compute Chebyshev Vandermonde for all points? 
    # V_cheb shape (n, 6)
    V_cheb_all = np.polynomial.chebyshev.chebvander(z, 5)
    y = training_subgrid_tendency
    
    for alpha in alphas:
        if _check_deadline():
            break
            
        cv_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            V_train = V_cheb_all[train_idx]
            y_train = y[train_idx]
            V_val = V_cheb_all[val_idx]
            y_val = y[val_idx]
            
            # Ridge Regression: min ||V c - y||^2 + alpha ||c||^2
            # (V^T V + alpha I) c = V^T y
            try:
                A = V_train.T @ V_train + alpha * np.eye(6)
                b = V_train.T @ y_train
                c_cheb = np.linalg.solve(A, b)
                
                # Evaluate on validation fold
                pred_val = V_val @ c_cheb
                mse_val = np.mean((pred_val - y_val) ** 2)
                cv_mses.append(mse_val)
            except Exception:
                cv_mses.append(np.inf)
                
        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_alpha = alpha
            
    # 3. Final Fit with best alpha on all data
    if _check_deadline():
        # Fallback to simple fit if running out of time
        try:
            c_cheb = np.polynomial.chebyshev.chebfit(z, y, 5)
        except Exception:
            c_cheb = np.zeros(6)
    else:
        A = V_cheb_all.T @ V_cheb_all + best_alpha * np.eye(6)
        b = V_cheb_all.T @ y
        try:
            c_cheb = np.linalg.solve(A, b)
        except Exception:
            c_cheb = np.polynomial.chebyshev.chebfit(z, y, 5)
            
    # 4. Convert Chebyshev coefficients to Monomial coefficients in z
    c_z = np.polynomial.chebyshev.cheb2poly(c_cheb)
    if len(c_z) < 6:
        c_z_full = np.zeros(6)
        c_z_full[:len(c_z)] = c_z
        c_z = c_z_full
    else:
        c_z = c_z[:6]
        
    # 5. Transform from z-space to x-space
    # z = a*x + b
    # a = 2 / x_range
    # b = -1 - 2*p_low / x_range
    a_val = 2.0 / x_range
    b_val = -1.0 - 2.0 * p_low / x_range
    
    # Transformation matrix T such that c_x = T @ c_z
    # T[j, i] is coeff of x^j in (a*x + b)^i
    T = np.zeros((6, 6))
    T[0, 0] = 1.0
    for i in range(1, 6):
        for j in range(i):
            T[j, i] = b_val * T[j, i-1]
            if j + 1 <= i:
                T[j+1, i] += a_val * T[j, i-1]
                
    c_original = T @ c_z
    
    # Ensure finite
    if not np.all(np.isfinite(c_original)):
        c_original = np.zeros(6)
        c_original[0] = np.mean(y)
        
    return c_original
# EVOLVE-BLOCK-END
