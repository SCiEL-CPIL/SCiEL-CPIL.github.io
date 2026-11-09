"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input x from [-2, 2] to z in [-1, 1] for stable Chebyshev evaluation
    z = x / 2.0
    
    # Precompute Chebyshev polynomials T_0(z) through T_9(z)
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
    
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    T5 = 2.0 * z * T4 - T3
    T6 = 2.0 * z * T5 - T4
    T7 = 2.0 * z * T6 - T5
    T8 = 2.0 * z * T7 - T6
    T9 = 2.0 * z * T8 - T7
    
    # Combine with parameters
    # params[0] * T0 + params[1] * T1 + ... + params[9] * T9
    return (params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3 +
            params[4] * T4 + params[5] * T5 + params[6] * T6 + params[7] * T7 +
            params[8] * T8 + params[9] * T9)
# EVOLVE-BLOCK-END
