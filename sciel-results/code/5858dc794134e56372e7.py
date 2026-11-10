"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use symmetry to minimize multiplications
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def _fit_centered_to_original(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        # Small diagonal for numerical safety
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Convert from centered/scaled basis to original basis
    # c_centered[k] is coeff for ((x-mu)/sigma)^k
    # We want coeff_orig[j] for x^j
    coeff_orig = np.zeros(degree + 1)
    
    for k in range(degree + 1):
        c_k = c_centered[k]
        if abs(c_k) < 1e-15:
            continue
            
        # Term is c_k * ((x-mu)/sigma)^k = c_k * sigma^(-k) * (x-mu)^k
        scale = c_k / (sigma ** k)
        
        # Expand (x-mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^(k-j)
        for j in range(k + 1):
            binom_val = _binom(k, j)
            mu_pow = (-mu) ** (k - j)
            coeff_orig[j] += scale * binom_val * mu_pow
            
    return coeff_orig

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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

    best_coeff = np.zeros(6)
    best_score = np.inf
    
    # We suspect that high degrees without strong regularization overfit.
    # The linear fit (degree 1) is the baseline (score 1.0).
    # We will search for a better fit using CV.
    
    n = len(x)
    
    # Prepare folds
    # Use a fixed seed for reproducibility of the split, but the data is fixed per call.
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Determine number of folds
    # If n is small, use fewer folds or leave-one-out?
    # Let's stick to 5 folds if n >= 20, else 3 folds.
    n_folds = 5 if n >= 20 else 3
    fold_size = n // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    # Ensure all indices are covered
    if len(folds[-1]) < fold_size:
        # Append remainder to last fold
        folds[-1] = np.concatenate([folds[-1], indices[(n_folds-1)*fold_size:]])
        # Remove empty folds if any
        folds = [f for f in folds if len(f) > 0]
        n_folds = len(folds)

    # Candidate hyperparameters
    # Degrees: 1, 2, 3, 4, 5
    # Alphas: logarithmic scale
    degrees = [1, 2, 3, 4, 5]
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    best_cv_score = np.inf
    best_params = (1, 0.0) # Default to linear, no reg (or small reg)
    
    # Optimization: If time is tight, reduce candidates
    # Let's estimate time per candidate.
    # Fitting is O(n * d^2). CV is n_folds * fit.
    # n is likely in thousands. d is <= 5. This is very fast.
    
    for deg in degrees:
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                try:
                    c_orig = _fit_centered_to_original(x_train, y_train, deg, alpha)
                    y_pred = np.polynomial.polynomial.polyval(x_val, c_orig)
                    mse = np.mean((y_val - y_pred)**2)
                    cv_errors.append(mse)
                except Exception:
                    cv_errors.append(np.inf)
            
            mean_cv_error = np.mean(cv_errors)
            
            if mean_cv_error < best_cv_score:
                best_cv_score = mean_cv_error
                best_params = (deg, alpha)
                
        if time.time() > deadline:
            break

    deg, alpha = best_params
    
    # Final fit on all data
    try:
        best_coeff = _fit_centered_to_original(x, y, deg, alpha)
        if len(best_coeff) < 6:
            padded = np.zeros(6)
            padded[:len(best_coeff)] = best_coeff
            best_coeff = padded
        else:
            best_coeff = best_coeff[:6]
    except Exception:
        # Fallback to linear
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    # Ensure finite
    if not np.all(np.isfinite(best_coeff)):
        try:
            best_coeff = np.zeros(6)
            best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except:
            best_coeff = np.zeros(6)
            
    return best_coeff
# EVOLVE-BLOCK-END
