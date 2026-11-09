"""Symbolic regression: recover hidden function via basis expansion fit by BFGS."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on [-3, 3].
    
    Maps x from [-3, 3] to t in [-1, 1], then evaluates Chebyshev polynomials
    T_0(t) through T_9(t). This provides a well-conditioned, complete basis
    for smooth functions on the interval.
    
    The model is linear in params, so BFGS converges to the global optimum.
    """
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    T = np.zeros((10, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination: sum of c_i * T_i(t)
    result = T @ params
    
    return result
