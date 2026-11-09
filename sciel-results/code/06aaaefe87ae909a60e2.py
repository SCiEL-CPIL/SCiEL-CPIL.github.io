"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 shifted Chebyshev polynomials of the first kind.
    
    The input x is in [-3, 3]. We map to t in [-1, 1] via t = x / 3.
    Then we compute T_0(t), T_1(t), ..., T_9(t) and return a linear combination
    with coefficients params[0] through params[9].
    
    Shifted Chebyshev polynomials are orthogonal with respect to the uniform
    distribution on [-1, 1], which matches our uniform sampling on [-3, 3].
    This provides excellent conditioning for the least-squares fit performed
    by the evaluator's BFGS optimizer.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.empty((10, len(t)), dtype=float)
    T[0, :] = 1.0
    T[1, :] = t
    
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # params is a 1D array of length 10
    # Return the linear combination: sum_{k=0}^{9} params[k] * T_k(t)
    result = np.zeros_like(t)
    for k in range(10):
        result += params[k] * T[k, :]
    
    return result
# EVOLVE-BLOCK-END
