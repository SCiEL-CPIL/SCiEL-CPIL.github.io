"""Symbolic regression with 1 input(s) - Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Chebyshev polynomial expansion of degree 9 (10 coefficients).
    
    The Chebyshev polynomials T_k(x) form a stable, nearly-orthogonal basis
    on [-1, 1]. We map the input from [-2, 2] to [-1, 1] via t = x/2,
    then evaluate T_0(t) through T_9(t) and take a linear combination
    with the 10 parameters.
    
    Chebyshev polynomials are computed via the recurrence:
      T_0(t) = 1
      T_1(t) = t
      T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    This basis is well-conditioned and avoids the numerical instability
    of high-degree monomial polynomials, while still being able to
    capture a wide range of smooth functions including oscillations.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Map from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9 via recurrence
    T = np.zeros((10, len(t)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Linear combination: y = sum_{k=0}^{9} params[k] * T_k(t)
    y = np.zeros_like(t)
    for k in range(10):
        y += params[k] * T[k, :]
    
    return y
# EVOLVE-BLOCK-END
