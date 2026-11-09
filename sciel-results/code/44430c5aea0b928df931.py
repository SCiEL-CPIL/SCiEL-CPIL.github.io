"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map input domain [-3, 3] to standard Chebyshev interval [-1, 1]
    t = x / 3.0
    
    # Precompute Chebyshev polynomials of degree 0 through 9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # Linear combination with 10 parameters
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * T5 +
              params[6] * T6 +
              params[7] * T7 +
              params[8] * T8 +
              params[9] * T9)
    
    # Safety check: if result is not finite, return zeros to avoid invalid score
    if not np.all(np.isfinite(result)):
        return np.zeros_like(result)
        
    return result
# EVOLVE-BLOCK-END
