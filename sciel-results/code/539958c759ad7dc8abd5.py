"""Rewritten program: degree-5 polynomial closure for unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved fast-scale forcing.
    
    Parameters:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        numpy array of 6 finite real coefficients in ascending powers of x
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Ensure finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Simple smoothing of y to reduce noise (as per knowledge base)
    # We sort by x, apply a moving average, then unsort.
    # This helps the polynomial fit the mean trend better.
    if n > 20:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        # Determine window size for moving average
        # Keep it small to preserve structure but smooth noise
        win_size = max(3, n // 100)
        if win_size % 2 == 0:
            win_size += 1
            
        # Apply moving average
        kernel = np.ones(win_size) / win_size
        y_smoothed = np.convolve(y_sorted, kernel, mode='same')
        
        # Unsort
        y_unsorted = np.empty_like(y_sorted)
        y_unsorted[order] = y_smoothed
        y = y_unsorted

    # Center x for numerical stability
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-12:
        x_std = 1.0

    xs = (x - x_mean) / x_std

    # Build Vandermonde in scaled variable
    # Shape must be (n, 6) for lstsq(A, b) where A is (M, N) and b is (M,)
    V = np.empty((n, 6), dtype=np.float64)
    V[:, 0] = 1.0
    V[:, 1] = xs
    V[:, 2] = xs * xs
    V[:, 3] = V[:, 2] * xs
    V[:, 4] = V[:, 3] * xs
    V[:, 5] = V[:, 4] * xs

    # Solve least squares: V @ c = y
    # Use SVD for robustness
    try:
        # Compute via lstsq
        c_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to normal equations
        try:
            VtV = V.T @ V
            Vty = V.T @ y
            c_scaled = np.linalg.solve(VtV + 1e-12 * np.eye(6), Vty)
        except np.linalg.LinAlgError:
            c_scaled = np.zeros(6, dtype=np.float64)

    # Convert from scaled variable back to original x
    # We have P(xs) = sum_{i=0}^{5} c_scaled[i] * xs^i
    # where xs = (x - x_mean) / x_std
    # We want P(x) = sum_{j=0}^{5} c_original[j] * x^j
    
    c_original = np.zeros(6, dtype=np.float64)
    
    for i in range(6):
        ci = c_scaled[i]
        if abs(ci) < 1e-15:
            continue
        
        # Expand ((x - x_mean) / x_std)^i
        # = (1/x_std^i) * (x - x_mean)^i
        # = (1/x_std^i) * sum_{j=0}^{i} C(i, j) * x^j * (-x_mean)^(i-j)
        
        inv_std_pow = (1.0 / x_std) ** i
        for j in range(i + 1):
            binom = comb(i, j)
            term = ci * binom * ((-x_mean) ** (i - j)) * inv_std_pow
            c_original[j] += term

    # Check for non-finite values
    if not np.all(np.isfinite(c_original)):
        c_original = np.nan_to_num(c_original, nan=0.0, posinf=0.0, neginf=0.0)

    return c_original
# EVOLVE-BLOCK-END
