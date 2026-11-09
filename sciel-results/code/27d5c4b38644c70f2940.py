"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 on domain [-3, 3].
    
    Maps x from [-3, 3] to t in [-1, 1] via t = x / 3.
    Uses the three-term recurrence for Chebyshev polynomials:
      T_0(t) = 1
      T_1(t) = t
      T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    params has length 10: coefficients c_0 ... c_9 for T_0 ... T_9.
    """
    # Map to [-1, 1]
    t = x / 3.0
    
    # Build Chebyshev polynomials T_0 through T_9 using recurrence
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    basis = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * t * basis[-1] - basis[-2]
        basis.append(Tn)
    
    # Combine with parameters
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * basis[i]
    
    return result
# EVOLVE-BLOCK-END
