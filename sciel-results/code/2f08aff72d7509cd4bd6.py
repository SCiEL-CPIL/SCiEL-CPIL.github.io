"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1]
    u = x / 3.0
    
    # Build a 10-term basis that is orthogonal (Chebyshev) and includes
    # some non-polynomial terms for richer expressiveness.
    # T_0 to T_9 are Chebyshev polynomials of the first kind.
    
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2.0 * u * T1 - T0
    T3 = 2.0 * u * T2 - T1
    T4 = 2.0 * u * T3 - T2
    T5 = 2.0 * u * T4 - T3
    T6 = 2.0 * u * T5 - T4
    T7 = 2.0 * u * T6 - T5
    T8 = 2.0 * u * T7 - T6
    T9 = 2.0 * u * T8 - T7
    
    # All 10 params weight Chebyshev polynomials T0 through T9.
    # Chebyshev basis on [-1,1] is well-conditioned (O(1) magnitude),
    # giving BFGS a near-orthogonal design matrix for fast convergence.
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4 + 
         params[5] * T5 + 
         params[6] * T6 + 
         params[7] * T7 +
         params[8] * T8 +
         params[9] * T9)
    
    return y
# EVOLVE-BLOCK-END
