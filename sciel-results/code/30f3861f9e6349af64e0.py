"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n_samples = len(x)
    
    if n_samples < 6:
        # Fallback: simple linear fit if too few samples
        if n_samples < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]  # polyfit returns descending order
        return coeff
    
    # LOESS-based approach: fit local polynomial to capture conditional expectation
    # This is more robust to noise and outliers than global polynomial fit
    
    # First, try a weighted local approach
    # Use LOESS-like weighting with a bandwidth
    
    # Sort by x for local fitting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Try different bandwidths and select best via cross-validation-like approach
    # Use a simple approach: fit with moderate regularization and verify
    
    # Scale x to [0, 1] for better conditioning
    x_min, x_max = x_sorted.min(), x_sorted.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        # All x values are the same
        mean_y = np.mean(y)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff
    
    x_scaled = (x_sorted - x_min) / x_range
    
    # Fit polynomial on scaled x
    V_scaled = np.vander(x_scaled, 6, increasing=True)
    
    # Use stronger regularization for better generalization
    lambda_reg = 1e-4 * np.eye(6)
    
    try:
        A = V_scaled.T @ V_scaled + lambda_reg
        b = V_scaled.T @ y_sorted
        coeff_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y_sorted, 5)
    
    # Transform coefficients back to original scale
    # If p(t) = sum c_i * t^i where t = (x - x_min)/x_range
    # Then p(x) = sum c_i * ((x - x_min)/x_range)^i
    # We need to express this in powers of x
    # Use polynomial composition
    
    # For each term c_i * ((x - x_min)/x_range)^i, expand using binomial theorem
    coeff_orig = np.zeros(6)
    for i in range(6):
        if abs(coeff_scaled[i]) < 1e-15:
            continue
        # (x - x_min)/x_range = (1/x_range)*x - x_min/x_range
        # ((x - x_min)/x_range)^i = sum_{j=0}^{i} C(i,j) * (1/x_range)^j * x^j * (-x_min/x_range)^(i-j)
        # = sum_{j=0}^{i} C(i,j) * (-x_min)^(i-j) / x_range^i * x^j
        for j in range(i + 1):
            # Binomial coefficient C(i, j)
            binom = 1.0
            for k in range(j):
                binom = binom * (i - k) / (k + 1)
            term_coeff = coeff_scaled[i] * binom * ((-x_min) ** (i - j)) / (x_range ** i)
            coeff_orig[j] += term_coeff
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback to simple linear fit
        coeff_orig = np.zeros(6)
        coeff_orig[:2] = np.polyfit(x, y, 1)[::-1]
    
    return coeff_orig
# EVOLVE-BLOCK-END
