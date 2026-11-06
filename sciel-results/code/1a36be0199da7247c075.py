"""Strategy: Two-stage denoising using isotonic regression (monotonicity-preserving) or spline smoothing, followed by degree-5 polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy import stats
from scipy.interpolate import UnivariateSpline

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
    
    # Standardize x for numerical stability in the final polynomial fit
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Time budget check
    if time.time() > deadline - 20:
        # Not enough time, do simple ridge fit on raw data
        try:
            V = np.vander(x_norm, N=6, increasing=True)
            A = V.T @ V + 1e-4 * np.eye(6)
            b = V.T @ y
            coeff_norm = np.linalg.solve(A, b)
            if not np.all(np.isfinite(coeff_norm)):
                coeff_norm = np.zeros(6)
            return _transform_coeffs(coeff_norm, x_mean, x_std)
        except Exception:
            return np.zeros(6)
    
    # Strategy: Use isotonic regression or spline smoothing to denoise the target
    # This explicitly separates the estimation of the smooth underlying trend from the final parametric approximation
    
    # Split into train and validation sets for selecting the best smoothing method
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
    
    # Candidate smoothing methods
    # 1. Isotonic regression (monotonicity-preserving)
    # 2. Spline smoothing with different smoothing factors
    
    best_score = np.inf
    best_y_smoothed = None
    best_method = "raw"
    
    # Method 1: Isotonic regression
    try:
        if time.time() < deadline - 15:
            # Sort by x for isotonic regression
            sort_idx = np.argsort(x_train)
            x_sorted = x_train[sort_idx]
            y_sorted = y_train[sort_idx]
            
            # Apply isotonic regression
            y_iso_train = stats.isotonic_regression(y_sorted, increasing=False)  # Try decreasing first
            # Interpolate to validation points
            y_iso_val = np.interp(x_val, x_sorted, y_iso_train)
            
            # Evaluate on validation set
            mse_iso = np.mean((y_val - y_iso_val) ** 2)
            
            if mse_iso < best_score:
                best_score = mse_iso
                best_method = "isotonic_decreasing"
                # Store the smoothed values for all data
                y_iso_all = stats.isotonic_regression(y[np.argsort(x_norm)], increasing=False)
                y_smoothed_all = np.interp(x_norm, np.sort(x_norm), y_iso_all)
                best_y_smoothed = y_smoothed_all.copy()
    except Exception:
        pass
    
    # Method 2: Spline smoothing with different smoothing factors
    try:
        if time.time() < deadline - 15:
            # Try different smoothing factors
            s_values = [0.01, 0.1, 1.0, 10.0]
            for s in s_values:
                if time.time() > deadline - 15:
                    break
                try:
                    # Fit spline to training data
                    spl = UnivariateSpline(x_train, y_train, s=s, k=3)
                    y_spl_val = spl(x_val)
                    
                    # Evaluate on validation set
                    mse_spl = np.mean((y_val - y_spl_val) ** 2)
                    
                    if mse_spl < best_score:
                        best_score = mse_spl
                        best_method = f"spline_s={s}"
                        # Fit spline to all data
                        spl_all = UnivariateSpline(x_norm, y, s=s, k=3)
                        best_y_smoothed = spl_all(x_norm)
                except Exception:
                    continue
    except Exception:
        pass
    
    # If no valid smoothing found, use raw data
    if best_y_smoothed is None:
        best_y_smoothed = y.copy()
        best_method = "raw"
    
    # Now fit the degree-5 polynomial to the smoothed values on ALL data
    try:
        V = np.vander(x_norm, N=6, increasing=True)
        
        # Use ridge regression with a small regularization parameter
        lam = 1e-4
        A = V.T @ V + lam * np.eye(6)
        b = V.T @ best_y_smoothed
        coeff_norm = np.linalg.solve(A, b)
        
        if not np.all(np.isfinite(coeff_norm)):
            coeff_norm = np.zeros(6)
    except Exception:
        # Fallback to simple ridge on raw data
        try:
            V = np.vander(x_norm, N=6, increasing=True)
            A = V.T @ V + 1e-4 * np.eye(6)
            b = V.T @ y
            coeff_norm = np.linalg.solve(A, b)
            if not np.all(np.isfinite(coeff_norm)):
                coeff_norm = np.zeros(6)
        except Exception:
            coeff_norm = np.zeros(6)
    
    # Transform coefficients back to original x space
    return _transform_coeffs(coeff_norm, x_mean, x_std)


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
