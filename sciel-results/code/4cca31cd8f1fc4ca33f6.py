"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic polynomial basis:
    p0: 1
    p1: x
    p2: y
    p3: z
    p4: x^2
    p5: y^2
    p6: z^2
    p7: xy
    p8: xz
    p9: yz
    """
    f0 = 1.0
    f1 = x
    f2 = y
    f3 = z
    f4 = x * x
    f5 = y * y
    f6 = z * z
    f7 = x * y
    f8 = x * z
    f9 = y * z
    
    pred = (params[0] * f0 + 
            params[1] * f1 + 
            params[2] * f2 + 
            params[3] * f3 + 
            params[4] * f4 + 
            params[5] * f5 + 
            params[6] * f6 + 
            params[7] * f7 + 
            params[8] * f8 + 
            params[9] * f9)
            
    return pred
# EVOLVE-BLOCK-END
