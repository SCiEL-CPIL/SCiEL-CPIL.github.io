"""Strategy: Polynomial basis fit with explicit conditioning and adaptive Tikhonov regularization, no kernel smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Construct design matrix for degree 5 polynomial in normalized space
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Time budget check
    if time.time() > deadline - 20:
        # Not enough time, do simple ridge fit
        try:
            A = V.T @ V + 1e-4 * np.eye(6)
            b = V.T @ y
            coeff_norm = np.linalg.solve(A, b)
            if not np.all(np.isfinite(coeff_norm)):
                coeff_norm = np.zeros(6)
            return _transform_coeffs(coeff_norm, x_mean, x_std)
        except Exception:
            return np.zeros(6)
    
    # Strategy: Use internal validation to select optimal regularization strength
    # This prevents overfitting on the training data while capturing the true signal
    
    # Split into train and validation sets
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_frac = min(0.2, max(0.1, 2000.0 / n))
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x_norm[train_idx]
    y_train = y[train_idx]
    x_val = x_norm[val_idx]
    y_val = y[val_idx]
    
    V_train = np.vander(x_train, N=6, increasing=True)
    V_val = np.vander(x_val, N=6, increasing=True)
    
    # Candidate regularization strengths
    lambda_candidates = np.logspace(-8, 4, 30)
    
    best_lambda = 1e-4
    best_score = np.inf
    best_coeff = None
    
    for lam in lambda_candidates:
        if time.time() > deadline - 10:
            break
        try:
            # Ridge regression
            A = V_train.T @ V_train + lam * np.eye(6)
            b = V_train.T @ y_train
            coeff_norm = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(coeff_norm)):
                continue
            
            # Evaluate on validation set
            y_pred = V_val @ coeff_norm
            mse = np.mean((y_val - y_pred) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_lambda = lam
                best_coeff = coeff_norm.copy()
        except Exception:
            continue
    
    # If no valid fit found, fall back to simple ridge
    if best_coeff is None:
        try:
            A = V.T @ V + 1e-4 * np.eye(6)
            b = V.T @ y
            best_coeff = np.linalg.solve(A, b)
            if not np.all(np.isfinite(best_coeff)):
                best_coeff = np.zeros(6)
        except Exception:
            best_coeff = np.zeros(6)
    
    # Now refit on ALL data with the best lambda
    try:
        A = V.T @ V + best_lambda * np.eye(6)
        b = V.T @ y
        best_coeff = np.linalg.solve(A, b)
        if not np.all(np.isfinite(best_coeff)):
            best_coeff = np.zeros(6)
    except Exception:
        pass
    
    # Transform coefficients back to original x space
    return _transform_coeffs(best_coeff, x_mean, x_std)


def _transform_coeffs(coeff_norm, x_mean, x_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    coeff_norm: coefficients for polynomial in x_norm = (x - x_mean)/x_std
    Returns: coefficients for polynomial in x (original space)
    """
    # We have: f(x) = sum_{i=0}^{5} c_i * ((x - mu)/sigma)^i
    # = sum_{i=0}^{5} c_i / sigma^i * (x - mu)^i
    # Now expand (x - mu)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-mu)^(i-j)
    # So f(x) = sum_{i=0}^{5} sum_{j=0}^{i} c_i / sigma^i * binom(i,j) * x^j * (-mu)^(i-j)
    # = sum_{j=0}^{5} x^j * sum_{i=j}^{5} c_i / sigma^i * binom(i,j) * (-mu)^(i-j)
    
    coeff_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom = _binom(i, j)
            s += coeff_norm[i] / (x_std ** i) * binom * ((-x_mean) ** (i - j))
        coeff_orig[j] = s
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig


def _binom(n, k):
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return float(result)
# EVOLVE-BLOCK-END
