# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses Ridge Regression on centered/scaled x to improve generalization
    and numerical stability, then converts coefficients back to original x basis.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle potential NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    # Center and scale x to improve conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for scaled x
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Ridge Regression: (V^T V + lambda * I) c = V^T y
    # Choose a small lambda for regularization
    VtV = V.T @ V
    trace_VtV = np.trace(VtV)
    lambda_reg = 1e-6 * trace_VtV / 6.0
    
    A = VtV + lambda_reg * np.eye(6)
    b = V.T @ y
    
    try:
        # Solve for coefficients in scaled basis
        c_scaled = np.linalg.solve(A, b)
        
        # Convert coefficients from scaled basis (x_s = (x - mu)/sigma) to original x basis
        # P(x_s) = sum c_scaled[i] * x_s^i
        # x_s = (x - mu) / sigma
        # P(x) = sum c_scaled[i] * sigma^-i * (x - mu)^i
        # We need to expand (x - mu)^i = sum_{k=0}^i binom(i, k) x^k (-mu)^(i-k)
        
        c_orig = np.zeros(6, dtype=np.float64)
        
        for i in range(6):
            coeff_i = c_scaled[i] * (1.0 / (x_std ** i))
            # Expand (x - mu)^i
            # (x - mu)^i = sum_{k=0}^i C(i, k) x^k (-mu)^(i-k)
            for k in range(i + 1):
                # Binomial coefficient C(i, k)
                binom = 1
                for m in range(k):
                    binom = binom * (i - m) / (m + 1)
                term = coeff_i * binom * ((-x_mean) ** (i - k))
                c_orig[k] += term
                
        # Check for finiteness
        if not np.all(np.isfinite(c_orig)):
            return np.zeros(6)
            
        return c_orig
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
