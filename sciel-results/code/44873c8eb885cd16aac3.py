"""Rewritten program using Chebyshev polynomial basis for improved stability and extrapolation."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses Chebyshev polynomial basis scaled to an adaptive range. Chebyshev
    polynomials are bounded on [-1, 1], which provides better stability for
    extrapolation to the unseen F=10 forcing regime compared to Legendre.
    """
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten inputs
    x = training_x.ravel()
    y = training_subgrid_tendency.ravel()
    
    # Remove any NaN/inf just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Fallback: return zeros if insufficient data
        return np.zeros(6)
    
    # Determine scaling parameters
    # We want to map the training data (and some margin for validation) to [-1, 1].
    # Let's use the min and max of the training data, but expand the range
    # to account for the unseen F=10 regime which likely has larger X values.
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Add a margin. The F=10 regime typically has ~20-30% larger variance.
    # Let's expand the range by 10% on each side to be safe, but not too much
    # to avoid ill-conditioning.
    range_val = x_max - x_min
    if range_val < 1e-10:
        range_val = 1.0
        
    # Expand range by 15% on each side
    margin = 0.15 * range_val
    x_min_exp = x_min - margin
    x_max_exp = x_max + margin
    
    # Center and scale to [-1, 1]
    mu = (x_max_exp + x_min_exp) / 2.0
    sigma = (x_max_exp - x_min_exp) / 2.0
    
    # Ensure sigma is not zero
    if sigma < 1e-10:
        sigma = 1.0
        
    # Transform x to t in [-1, 1] (approximately for training data)
    t = (x - mu) / sigma
    
    # Build Chebyshev polynomial design matrix of degree 5
    n = len(t)
    A = np.zeros((n, 6))
    
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    A[:, 0] = 1.0
    # T_1(t) = t
    A[:, 1] = t
    # T_2(t) = 2t^2 - 1
    A[:, 2] = 2.0 * t**2 - 1.0
    # T_3(t) = 4t^3 - 3t
    A[:, 3] = 4.0 * t**3 - 3.0 * t
    # T_4(t) = 8t^4 - 8t^2 + 1
    A[:, 4] = 8.0 * t**4 - 8.0 * t**2 + 1.0
    # T_5(t) = 16t^5 - 20t^3 + 5t
    A[:, 5] = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Solve least squares for Chebyshev coefficients
    try:
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simple linear fit
        coeff = np.zeros(6)
        try:
            c1, c0 = np.polyfit(x, y, 1)
            coeff[0] = c0
            coeff[1] = c1
        except Exception:
            coeff[0] = np.mean(y) if len(y) > 0 else 0.0
        return coeff
    
    # Convert Chebyshev coefficients to monomial coefficients
    # f(x) = sum_k a_k T_k( (x - mu) / sigma )
    # We need c_i such that f(x) = sum_i c_i x^i
    
    # t_coeffs[k][i] is the coefficient of t^i in T_k(t)
    t_coeffs = np.zeros((6, 6))
    t_coeffs[0, 0] = 1.0
    t_coeffs[1, 1] = 1.0
    t_coeffs[2, 0] = -1.0
    t_coeffs[2, 2] = 2.0
    t_coeffs[3, 1] = -3.0
    t_coeffs[3, 3] = 4.0
    t_coeffs[4, 0] = 1.0
    t_coeffs[4, 2] = -8.0
    t_coeffs[4, 4] = 8.0
    t_coeffs[5, 1] = 5.0
    t_coeffs[5, 3] = -20.0
    t_coeffs[5, 5] = 16.0
    
    # Coefficients in t = (x - mu) / sigma
    # g(t) = sum_k a_k T_k(t) = sum_i b_i t^i
    b = np.zeros(6)
    for i in range(6):
        b[i] = np.sum(cheb_coeffs * t_coeffs[:, i])
    
    # Now substitute t = (x - mu) / sigma
    # g(t) = sum_i b_i ( (x - mu) / sigma )^i = sum_i (b_i / sigma^i) (x - mu)^i
    # (x - mu)^i = sum_{m=0}^i binom(i, m) x^m (-mu)^(i-m)
    
    from math import comb
    
    monomial_coeffs = np.zeros(6)
    for i in range(6):
        if b[i] == 0:
            continue
        coeff_i = b[i] / (sigma ** i)
        for m in range(i + 1):
            monomial_coeffs[m] += coeff_i * comb(i, m) * ((-mu) ** (i - m))
            
    # Clip very small values to zero for cleanliness
    monomial_coeffs[np.abs(monomial_coeffs) < 1e-15] = 0.0
    
    return monomial_coeffs
# EVOLVE-BLOCK-END
