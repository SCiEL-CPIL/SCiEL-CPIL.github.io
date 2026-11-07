"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a full quadratic polynomial basis.
    The basis consists of 10 terms:
    1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    This allows the optimizer to fit any function that can be approximated by a quadratic surface.
    """
    p = params
    
    # Pre-compute basis terms
    # t0: 1
    # t1: x
    # t2: y
    # t3: z
    # t4: x^2
    # t5: y^2
    # t6: z^2
    # t7: xy
    # t8: xz
    # t9: yz
    
    t1 = x
    t2 = y
    t3 = z
    t4 = x * x
    t5 = y * y
    t6 = z * z
    t7 = x * y
    t8 = x * z
    t9 = y * z
    
    # Linear combination
    # p[0] is the constant term (intercept)
    result = (p[0] * np.ones_like(x) + 
              p[1] * t1 + 
              p[2] * t2 + 
              p[3] * t3 + 
              p[4] * t4 + 
              p[5] * t5 + 
              p[6] * t6 + 
              p[7] * t7 + 
              p[8] * t8 + 
              p[9] * t9)
              
    return result
# EVOLVE-BLOCK-END
