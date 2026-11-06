"""Combined approach: Nadaraya-Watson denoising + orthogonal Legendre polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson(x, y, x_eval, bandwidth):
    """Nadaraya-Watson kernel regression estimator (vectorized in chunks)."""
    if bandwidth <= 0:
        return np.full_like(x_eval, np.mean(y), dtype=np.float64)
    n_eval = len(x_eval)
    result = np.empty(n_eval, dtype=np.float64)
    chunk_size = max(1, min(2000, n_eval))
    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        xe = x_eval[start:end]
        diff = xe[:, None] - x[None, :]
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        num = weights @ y
        den = weights.sum(axis=1)
        den_safe = np.where(den > 1e-30, den, 1.0)
        result[start:end] = num / den_safe
    return result


def _legendre_basis(x, x_min, x_max, degree):
    """
    Build scaled Legendre polynomial basis matrix.
    Maps x from [x_min, x_max] to t in [-1, 1], then evaluates Legendre P_i(t).
    Returns (n, degree+1) matrix.
    """
    n = len(x)
    if x_max - x_min < 1e-15:
        # Degenerate: return monomials (just ones for constant)
        V = np.ones((n, degree + 1))
        return V
    t = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    # Clip to [-1, 1] to avoid numerical issues at boundaries
    t = np.clip(t, -1.0, 1.0)
    # Recursively compute Legendre polynomials
    P = np.zeros((n, degree + 1), dtype=np.float64)
    P[:, 0] = 1.0
    if degree >= 1:
        P[:, 1] = t
    for k in range(2, degree + 1):
        P[:, k] = ((2 * k - 1) * t * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    return P


def _fit_polynomial_orthogonal(x, y, x_min, x_max, degree=5, ridge_lambda=1e-8):
    """
    Fit degree-5 polynomial using orthogonal Legendre basis.
    Returns coefficients in monomial basis (ascending powers).
    """
    n = len(x)
    if n < degree + 1:
        # Not enough points; fall back to lower degree
        deg = max(1, n - 1)
        V = _legendre_basis(x, x_min, x_max, deg)
        A = V.T @ V + ridge_lambda * np.eye(deg + 1)
        b_vec = V.T @ y
        try:
            c_orth = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            c_orth = np.linalg.lstsq(V, y, rcond=None)[0]
        # Convert from Legendre basis to monomial basis
        # We need to express sum c_i * P_i(t(x)) as sum d_j * x^j
        # This requires converting Legendre polynomials to monomials
        return _legendre_to_monomial(c_orth, x_min, x_max, deg)
    
    V = _legendre_basis(x, x_min, x_max, degree)
    A = V.T @ V + ridge_lambda * np.eye(degree + 1)
    b_vec = V.T @ y
    try:
        c_orth = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        c_orth = np.linalg.lstsq(V, y, rcond=None)[0]
    
    return _legendre_to_monomial(c_orth, x_min, x_max, degree)


def _legendre_to_monomial(c_orth, x_min, x_max, degree):
    """
    Convert Legendre-basis coefficients to monomial basis coefficients.
    P_k(t) where t = 2(x - x_min)/(x_max - x_min) - 1
    We need: sum_k c_k * P_k(t(x)) = sum_j d_j * x^j
    """
    # Build the conversion matrix: express each P_k(t(x)) as a polynomial in x
    # t = a*x + b where a = 2/(x_max-x_min), b = -2*x_min/(x_max-x_min) - 1
    if x_max - x_min < 1e-15:
        # Constant case
        result = np.zeros(degree + 1)
        result[0] = np.sum(c_orth[:degree + 1])  # All P_k(0) evaluated... actually just mean
        return result
    
    a = 2.0 / (x_max - x_min)
    b = -2.0 * x_min / (x_max - x_min) - 1.0
    
    # Compute Legendre polynomials in t, then substitute t = a*x + b
    # P_k(t) is a polynomial in t of degree k
    # We need to express it as a polynomial in x
    
    # Build monomial coefficients for each P_k(t(x))
    # P_0(t) = 1
    # P_1(t) = t = a*x + b
    # P_k(t) = ((2k-1)*t*P_{k-1}(t) - (k-1)*P_{k-2}(t)) / k
    
    # Represent each polynomial as a list of monomial coefficients (ascending)
    polys = []  # polys[k] = array of length k+1, coefficients of x^0, x^1, ..., x^k
    
    # P_0
    polys.append(np.array([1.0]))
    
    if degree >= 1:
        # P_1(t) = t = a*x + b
        polys.append(np.array([b, a]))
    
    for k in range(2, degree + 1):
        # P_k(t) = ((2k-1)*t*P_{k-1}(t) - (k-1)*P_{k-2}(t)) / k
        # t * P_{k-1}(t): multiply P_{k-1} polynomial (in x) by (a*x + b)
        p_prev = polys[k - 1]  # length k
        p_prev2 = polys[k - 2]  # length k-1
        
        # t * P_{k-1}: result has degree k, length k+1
        t_times_prev = np.zeros(k + 1)
        # Multiply by a*x: shift up by 1
        t_times_prev[1:] += a * p_prev
        # Multiply by b: same degree
        t_times_prev[:k] += b * p_prev
        
        # (k-1) * P_{k-2}: length k-1, pad to k+1
        k_minus_1_p_prev2 = np.zeros(k + 1)
        k_minus_1_p_prev2[:k - 1] = (k - 1) * p_prev2
        
        p_k = ((2 * k - 1) * t_times_prev - k_minus_1_p_prev2) / k
        polys.append(p_k)
    
    # Now sum c_k * P_k(x)
    result = np.zeros(degree + 1)
    for k in range(degree + 1):
        result += c_orth[k] * polys[k]
    
    return result


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Basic sanity checks
    if n < 6:
        return np.zeros(6)
    
    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    x_min = x_sorted[0]
    x_max = x_sorted[-1]
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        coeff = np.zeros(6)
        coeff[0] = np.mean(y_sorted)
        return coeff
    
    # Strategy: Try Nadaraya-Watson smoothing with different bandwidths,
    # then fit degree-5 polynomial in orthogonal Legendre basis.
    # Use cross-validation to select best bandwidth.
    
    # Create evaluation grid for smoothing
    n_grid = min(400, max(80, n // 50))
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    # Base bandwidth
    base_bw = x_range * 0.05
    bandwidths = [base_bw * 0.3, base_bw * 0.5, base_bw, base_bw * 1.5, base_bw * 2.0, base_bw * 3.0]
    
    # Cross-validation: 5-fold
    n_folds = 5
    fold_size = n // n_folds
    
    best_coeff = None
    best_cv_error = np.inf
    
    for bw in bandwidths:
        if time.time() > deadline - 15:
            break
        
        # Cross-validate
        cv_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_start:val_end] = False
            
            x_train = x_sorted[train_mask]
            y_train = y_sorted[train_mask]
            x_val = x_sorted[val_start:val_end]
            y_val = y_sorted[val_start:val_end]
            
            if len(x_train) < 15 or len(x_val) < 5:
                continue
            
            # Smooth training data at validation points
            y_smooth_val = _nadaraya_watson(x_train, y_train, x_val, bw)
            
            # Fit orthogonal polynomial to smoothed data
            x_min_cv = x_train[0]
            x_max_cv = x_train[-1]
            if x_max_cv - x_min_cv < 1e-12:
                continue
            
            try:
                coeff_cv = _fit_polynomial_orthogonal(x_val, y_smooth_val, x_min_cv, x_max_cv, degree=5)
            except Exception:
                continue
            
            if not np.all(np.isfinite(coeff_cv)):
                continue
            
            # Evaluate on validation set
            pred = np.polynomial.polynomial.polyval(x_val, coeff_cv)
            mse = np.mean((pred - y_val) ** 2)
            cv_errors.append(mse)
        
        if cv_errors:
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                # Save the bandwidth and fit on all data
                # We'll refit on all data later
                best_bw = bw
    
    if time.time() > deadline - 5:
        if best_coeff is not None:
            return best_coeff
        # Fallback: direct orthogonal fit on all data
        try:
            coeff = _fit_polynomial_orthogonal(x_sorted, y_sorted, x_min, x_max, degree=5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)
    
    # Final fit on all data using best bandwidth (if found) or direct fit
    if best_cv_error < np.inf and 'best_bw' in dir():
        # Smooth all data at grid points, then fit
        y_smooth_grid = _nadaraya_watson(x_sorted, y_sorted, x_grid, best_bw)
        try:
            coeff = _fit_polynomial_orthogonal(x_grid, y_smooth_grid, x_min, x_max, degree=5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
    
    # Fallback: direct orthogonal fit on all data
    try:
        coeff = _fit_polynomial_orthogonal(x_sorted, y_sorted, x_min, x_max, degree=5)
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass
    
    # Final fallback: linear fit
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    except Exception:
        return np.zeros(6)

# EVOLVE-BLOCK-END
