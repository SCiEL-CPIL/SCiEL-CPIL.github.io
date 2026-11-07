"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 0-9 on [-2, 2].
    This provides a stable, orthogonal basis for smooth functions.
    """
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Combine with parameters
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * T[i, :]
    
    return result
# EVOLVE-BLOCK-END
