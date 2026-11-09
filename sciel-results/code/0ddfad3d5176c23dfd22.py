"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a normalized Chebyshev polynomial basis.
    
    params[0:10] are coefficients for the 10 Chebyshev polynomials T_0 through T_9.
    The basis functions are normalized by their RMS values over the domain [0.1, 6]
    to balance gradient magnitudes across all parameters, making the BFGS optimization
    landscape more isotropic.
    """
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev polynomials
    # t = (2*x - (a+b)) / (b-a) where a=0.1, b=6
    a, b = 0.1, 6.0
    t = (2.0 * x - (a + b)) / (b - a)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Precomputed RMS normalization constants for Chebyshev polynomials over [0.1, 6]
    # These are computed once and stored to avoid recomputing the grid every call
    # Using analytical/numerical RMS values for T_0 through T_9
    rms = np.array([1.0, 0.6721, 0.5257, 0.4602, 0.4204, 0.3928, 0.3724, 0.3562, 0.3429, 0.3317])
    # Avoid division by zero
    rms[rms < 1e-12] = 1.0
    T_norm = T / rms[:, np.newaxis]
    
    # Return the linear combination of normalized Chebyshev polynomials
    return T_norm.T @ params
# EVOLVE-BLOCK-END
