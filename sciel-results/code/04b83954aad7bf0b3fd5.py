"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomial basis of total degree 3
    # This basis is better conditioned than monomials for BFGS optimization.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    
    T0x = 1.0
    T1x = x
    T2x = 2.0 * x**2 - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    
    T0y = 1.0
    T1y = y
    T2y = 2.0 * y**2 - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    
    # Total degree 3 terms:
    # deg 0: T0x T0y
    # deg 1: T1x T0y, T0x T1y
    # deg 2: T2x T0y, T1x T1y, T0x T2y
    # deg 3: T3x T0y, T2x T1y, T1x T2y, T0x T3y
    
    return (p0 * T0x * T0y + 
            p1 * T1x * T0y + p2 * T0x * T1y + 
            p3 * T2x * T0y + p4 * T1x * T1y + p5 * T0x * T2y + 
            p6 * T3x * T0y + p7 * T2x * T1y + p8 * T1x * T2y + p9 * T0x * T3y)
# EVOLVE-BLOCK-END
