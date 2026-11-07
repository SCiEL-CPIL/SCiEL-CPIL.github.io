"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Legendre polynomial series with scaling x/3 to map [-3, 3] to [-1, 1].
    Basis: P_0(u), P_1(u), ..., P_9(u) where u = x/3.
    This basis is orthogonal on [-1, 1], leading to a well-conditioned linear system
    that BFGS can solve efficiently. It is particularly effective if the target
    function is a polynomial or smooth non-periodic function.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for safety
    if x.ndim != 1:
        x = x.flatten()
        
    # Scale x to [-1, 1]
    u = x / 3.0
    
    # Generate Legendre polynomials P_0 to P_9
    # P_n(u)
    p = np.zeros((10, len(u)))
    p[0] = 1.0
    if len(u) > 0:
        p[1] = u
        for n in range(1, 9):
            # Recurrence: (n+1)P_{n+1} = (2n+1)uP_n - nP_{n-1}
            p[n+1] = ((2 * n + 1) * u * p[n] - n * p[n-1]) / (n + 1)
            
    # Combine with parameters
    # result = sum(params[i] * P_i(u))
    result = np.dot(p.T, params)
    
    return result
# EVOLVE-BLOCK-END
