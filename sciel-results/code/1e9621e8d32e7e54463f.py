"""Symbolic regression using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Map input domain [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
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
    
    # Linear combination of basis functions
    # This is a linear model in params, which BFGS handles well
    result = (p[0] * T0 + 
              p[1] * T1 + 
              p[2] * T2 + 
              p[3] * T3 + 
              p[4] * T4 + 
              p[5] * T5 + 
              p[6] * T6 + 
              p[7] * T7 + 
              p[8] * T8 + 
              p[9] * T9)
              
    return result
# EVOLVE-BLOCK-END
