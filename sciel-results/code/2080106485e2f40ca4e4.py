"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a Chebyshev polynomial basis of degree 9.
    
    The Chebyshev basis is nearly orthogonal on [-1, 1], providing superior numerical
    conditioning compared to monomials. This stabilizes BFGS optimization for high-order
    polynomial targets and provides efficient approximation of smooth functions on bounded intervals.
    
    x is mapped from [0.1, 6] to [-1, 1] to leverage the Chebyshev polynomials' properties.
    """
    # Map x from [0.1, 6] to [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Ensure t is within [-1, 1] to avoid issues (though x should be in range)
    t = np.clip(t, -1.0, 1.0)
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    n = len(x)
    T = np.zeros((n, 10), dtype=np.float64)
    T[:, 0] = 1.0
    if n > 0:
        T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Compute linear combination: sum_{k=0}^{9} params[k] * T_k(t)
    return T @ params
# EVOLVE-BLOCK-END
