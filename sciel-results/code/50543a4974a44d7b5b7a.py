"""Rewritten program using orthogonal polynomial basis (Legendre) with adaptive scaling for extrapolation."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses Legendre polynomial basis scaled to an adaptive range that covers
    the training data and likely validation data (F=10) to improve
    generalization.
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
    # Center at the mean of the training data
    mu = np.mean(x)
    
    # Estimate scale
    # Use standard deviation but expand it to cover the unseen F=10 regime
    # which likely has a wider distribution of X.
    std_x = np.std(x)
    min_scale = 1.0
    range_half = (np.max(x) - np.min(x)) / 2.0
    
    # We want the training data to be well within [-1, 1] to allow some room for extrapolation
    # without hitting the "edge" of the orthogonal domain too hard.
    # If range_half is 8, sigma=10 means x in [-8,8] maps to [-0.8, 0.8].
    # Validation x in [-12,12] maps to [-1.2, 1.2]. This is reasonable.
    sigma = max(std_x * 1.5, range_half * 1.2, min_scale)
    
    # Normalize x
    x_norm = (x - mu) / sigma
    
    # Build Legendre polynomial design matrix of degree 5
    n = len(x_norm)
    A = np.zeros((n, 6))
    A[:, 0] = 1.0
    A[:, 1] = x_norm
    A[:, 2] = 0.5 * (3.0 * x_norm**2 - 1.0)
    A[:, 3] = 0.5 * (5.0 * x_norm**3 - 3.0 * x_norm)
    A[:, 4] = 0.125 * (35.0 * x_norm**4 - 30.0 * x_norm**2 + 3.0)
    A[:, 5] = 0.125 * (63.0 * x_norm**5 - 70.0 * x_norm**3 + 15.0 * x_norm)
    
    # Solve regularized least squares (Ridge) for Legendre coefficients
    # Regularization helps generalization to the unseen F=10 regime by preventing
    # high-order terms from overfitting to noise in the training data.
    try:
        # Compute Gram matrix and vector
        G = A.T @ A
        b_vec = A.T @ y
        
        # Add regularization
        # Lambda is chosen relative to the magnitude of the Gram matrix
        lambda_reg = 1e-4 * np.trace(G) / 6.0
        G_reg = G + lambda_reg * np.eye(6)
        
        legendre_coeffs = np.linalg.solve(G_reg, b_vec)
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
    
    # Convert Legendre coefficients to monomial coefficients
    # f(x) = sum_k a_k P_k( (x - mu) / sigma )
    # We need c_i such that f(x) = sum_i c_i x^i
    
    # p_coeffs[k][i] is the coefficient of t^i in P_k(t)
    p_coeffs = np.zeros((6, 6))
    p_coeffs[0, 0] = 1.0
    p_coeffs[1, 1] = 1.0
    p_coeffs[2, 0] = -0.5
    p_coeffs[2, 2] = 1.5
    p_coeffs[3, 1] = -1.5
    p_coeffs[3, 3] = 2.5
    p_coeffs[4, 0] = 0.375
    p_coeffs[4, 2] = -3.75
    p_coeffs[4, 4] = 4.375
    p_coeffs[5, 1] = 1.875
    p_coeffs[5, 3] = -8.75
    p_coeffs[5, 5] = 7.875
    
    # Coefficients in t = (x - mu) / sigma
    # g(t) = sum_k a_k P_k(t) = sum_i b_i t^i
    b = np.zeros(6)
    for i in range(6):
        b[i] = np.sum(legendre_coeffs * p_coeffs[:, i])
    
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
