import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial expansion of degree 9.
    
    Uses all 10 parameters as coefficients for T_0(x) through T_9(x).
    Chebyshev polynomials provide a stable, orthogonal-like global basis that
    efficiently captures high-frequency oscillations and complex smooth shapes
    without the numerical instability of high-degree monomial polynomials.
    """
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev domain
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    T = np.zeros((10, len(t)))
    T[0] = 1.0
    T[1] = t
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
    
    # Linear combination of Chebyshev polynomials
    y = np.zeros_like(t)
    for k in range(10):
        y += params[k] * T[k]
    
    return y
