"""Symbolic regression with Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a 9th-degree Chebyshev polynomial basis on x.
    
    Maps x from [-3, 3] to t in [-1, 1] via t = x / 3.
    Uses the 10 parameters as coefficients for T_0(t) through T_9(t).
    
    Chebyshev polynomials are computed via the three-term recurrence:
        T_0(t) = 1
        T_1(t) = t
        T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # map [-3, 3] -> [-1, 1]
    
    # T_0
    T0 = np.ones_like(t)
    result = params[0] * T0
    
    if len(t) == 0:
        return result
    
    # T_1
    T1 = t.copy()
    result = result + params[1] * T1
    
    # T_2 through T_9
    for n in range(2, 10):
        Tn = 2.0 * t * T1 - T0
        result = result + params[n] * Tn
        T0 = T1
        T1 = Tn
    
    return result
# EVOLVE-BLOCK-END
