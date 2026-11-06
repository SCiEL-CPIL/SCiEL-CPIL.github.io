"""Rewritten program using orthogonal polynomial basis (Legendre) for stable degree-5 fit."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses Legendre polynomial basis scaled to the training data range to
    eliminate multicollinearity, then converts back to monomial coefficients.
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
    
    # Scale x to [-1, 1] range for Legendre polynomial stability
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x values are the same
        # Fit a constant (degree 0) effectively
        coeff = np.zeros(6)
        coeff[0] = np.mean(y) if len(y) > 0 else 0.0
        return coeff
    
    # Normalize x to [-1, 1]
    x_norm = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)
    
    # Build Legendre polynomial design matrix of degree 5
    # P_0(x) = 1
    # P_1(x) = x
    # P_2(x) = (3x^2 - 1)/2
    # P_3(x) = (5x^3 - 3x)/2
    # P_4(x) = (35x^4 - 30x^2 + 3)/8
    # P_5(x) = (63x^5 - 70x^3 + 15x)/8
    
    n = len(x_norm)
    A = np.zeros((n, 6))
    A[:, 0] = 1.0
    A[:, 1] = x_norm
    A[:, 2] = 0.5 * (3.0 * x_norm**2 - 1.0)
    A[:, 3] = 0.5 * (5.0 * x_norm**3 - 3.0 * x_norm)
    A[:, 4] = 0.125 * (35.0 * x_norm**4 - 30.0 * x_norm**2 + 3.0)
    A[:, 5] = 0.125 * (63.0 * x_norm**5 - 70.0 * x_norm**3 + 15.0 * x_norm)
    
    # Solve least squares for Legendre coefficients
    try:
        legendre_coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
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
    # We need to express sum_{k=0}^{5} a_k * P_k(x_norm) as sum_{i=0}^{5} c_i * x^i
    # where x is the original variable, and x_norm = (x - mu) / sigma
    # mu = (x_min + x_max) / 2, sigma = (x_max - x_min) / 2
    
    mu = (x_min + x_max) / 2.0
    sigma = x_range / 2.0
    
    # First, express each P_k(x_norm) as a polynomial in x_norm:
    # P_0 = 1
    # P_1 = x_norm
    # P_2 = (3*x_norm^2 - 1)/2 = 1.5*x_norm^2 - 0.5
    # P_3 = (5*x_norm^3 - 3*x_norm)/2 = 2.5*x_norm^3 - 1.5*x_norm
    # P_4 = (35*x_norm^4 - 30*x_norm^2 + 3)/8 = 4.375*x_norm^4 - 3.75*x_norm^2 + 0.375
    # P_5 = (63*x_norm^5 - 70*x_norm^3 + 15*x_norm)/8 = 7.875*x_norm^5 - 8.75*x_norm^3 + 1.875*x_norm
    
    # Coefficients of P_k in powers of x_norm (ascending)
    # p_coeffs[k][i] = coefficient of x_norm^i in P_k
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
    
    # Now the polynomial in x_norm is: f(x_norm) = sum_k legendre_coeffs[k] * P_k(x_norm)
    # = sum_k legendre_coeffs[k] * sum_i p_coeffs[k][i] * x_norm^i
    # = sum_i (sum_k legendre_coeffs[k] * p_coeffs[k][i]) * x_norm^i
    
    # Coefficients in x_norm (ascending powers)
    xnorm_coeffs = np.zeros(6)
    for i in range(6):
        xnorm_coeffs[i] = np.sum(legendre_coeffs * p_coeffs[:, i])
    
    # Now substitute x_norm = (x - mu) / sigma
    # x_norm^i = (x - mu)^i / sigma^i
    # (x - mu)^i = sum_{j=0}^{i} binom(i, j) * x^j * (-mu)^(i-j)
    
    # So the coefficient of x^j in the final polynomial is:
    # c_j = sum_{i=j}^{5} xnorm_coeffs[i] / sigma^i * sum_{k=0}^{i-j} binom(i, k) * (-mu)^k * ... 
    # Actually: (x - mu)^i = sum_{m=0}^{i} C(i,m) * x^m * (-mu)^(i-m)
    # So coefficient of x^m from term i is: xnorm_coeffs[i] / sigma^i * C(i,m) * (-mu)^(i-m)
    
    from math import comb
    
    monomial_coeffs = np.zeros(6)
    for i in range(6):
        if xnorm_coeffs[i] == 0:
            continue
        for m in range(i + 1):
            # Coefficient of x^m from (x - mu)^i / sigma^i
            # (x - mu)^i = sum_{m=0}^{i} C(i,m) x^m (-mu)^(i-m)
            monomial_coeffs[m] += xnorm_coeffs[i] * comb(i, m) * ((-mu) ** (i - m)) / (sigma ** i)
    
    # Clip very small values to zero for cleanliness
    monomial_coeffs[np.abs(monomial_coeffs) < 1e-15] = 0.0
    
    return monomial_coeffs
# EVOLVE-BLOCK-END
