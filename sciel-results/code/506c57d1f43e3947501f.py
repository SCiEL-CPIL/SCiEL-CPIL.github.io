# EVOLVE-BLOCK-START
import numpy as np

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation using Chebyshev basis.
    
    Args:
        training_x: array-like, flattened slow-state values
        training_subgrid_tendency: array-like, flattened subgrid tendency values
    
    Returns:
        np.ndarray of shape (6,) with coefficients in ASCENDING powers of x
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Scale x to [-1, 1] for Chebyshev approximation
    # Using min/max ensures full coverage, but robust scaling might be better if outliers exist.
    # For Lorenz models, min/max is usually safe and ensures the domain is covered.
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min < 1e-12:
        # Constant x, return zero polynomial
        return np.zeros(6)
        
    # Map x to t in [-1, 1]
    # t = (2 * (x - x_min) / (x_max - x_min)) - 1
    # Inverse: x = x_min + (t + 1) * (x_max - x_min) / 2
    # Let's define the transformation explicitly for coefficient conversion later.
    # t = a * x + b
    a = 2.0 / (x_max - x_min)
    b = -1.0 - a * x_min
    
    t = a * x + b
    
    # Construct Chebyshev design matrix
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    deg = 5
    V = np.zeros((len(t), deg + 1))
    V[:, 0] = 1.0
    if deg >= 1:
        V[:, 1] = t
    for k in range(2, deg + 1):
        V[:, k] = 2.0 * t * V[:, k-1] - V[:, k-2]
        
    # Solve least squares: V c = y
    # Use lstsq for robustness
    try:
        c_cheb, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(6)
        
    # Convert Chebyshev coefficients to Monomial coefficients
    # We need to express T_k(t) in terms of x.
    # T_k(t) is a polynomial in t.
    # t = a*x + b.
    # So T_k(t) is a polynomial in x.
    
    # First, find the monomial coefficients of T_k(t) in terms of t.
    # Let M[k][j] be the coefficient of t^j in T_k(t).
    M = np.zeros((deg + 1, deg + 1))
    M[0, 0] = 1.0
    if deg >= 1:
        M[1, 1] = 1.0
    for k in range(2, deg + 1):
        # T_k(t) = 2t T_{k-1}(t) - T_{k-2}(t)
        # 2t T_{k-1}(t) shifts coefficients of T_{k-1} up by 1 and multiplies by 2
        M[k, 1:] = 2.0 * M[k-1, :-1]
        M[k, :] -= M[k-2, :]
        
    # Now, we have T_k(t) = sum_{j=0}^k M[k][j] * t^j
    # Substitute t = a*x + b
    # t^j = (a*x + b)^j = sum_{m=0}^j comb(j, m) * a^m * b^(j-m) * x^m
    
    from math import comb
    
    coeff_monomial = np.zeros(deg + 1)
    
    for k in range(deg + 1):
        c_k = c_cheb[k]
        if abs(c_k) < 1e-15:
            continue
            
        # Sum over j (power of t)
        for j in range(k + 1):
            m_k_j = M[k][j]
            if abs(m_k_j) < 1e-15:
                continue
                
            # Expand (a*x + b)^j
            for m in range(j + 1):
                coeff_monomial[m] += c_k * m_k_j * comb(j, m) * (a ** m) * (b ** (j - m))
                
    # Clip to finite values
    coeff_monomial = np.where(np.isfinite(coeff_monomial), coeff_monomial, 0.0)
    
    return coeff_monomial
# EVOLVE-BLOCK-END
