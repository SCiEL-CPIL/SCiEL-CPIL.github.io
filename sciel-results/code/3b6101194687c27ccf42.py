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
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Build up to T_9
    T = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * T[n-1] - T[n-2]
        T.append(Tn)
    
    # Combine with parameters
    y = params[0] * T[0] + params[1] * T[1] + params[2] * T[2] + params[3] * T[3] + \
        params[4] * T[4] + params[5] * T[5] + params[6] * T[6] + params[7] * T[7] + \
        params[8] * T[8] + params[9] * T[9]
    
    return y
