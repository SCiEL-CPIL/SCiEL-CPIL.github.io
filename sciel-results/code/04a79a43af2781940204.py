"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a polynomial basis.
    Basis terms: 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    This basis is chosen to capture diagonal cubic curvature, which is a common
    feature in hidden functions that are not purely quadratic.
    """
    p = params
    
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    
    # Basis order:
    # 0: 1
    # 1: x
    # 2: y
    # 3: z
    # 4: x^2
    # 5: y^2
    # 6: z^2
    # 7: x^3
    # 8: y^3
    # 9: z^3
    
    return (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * z + 
            p[4] * x2 + 
            p[5] * y2 + 
            p[6] * z2 + 
            p[7] * x3 + 
            p[8] * y3 + 
            p[9] * z3)
# EVOLVE-BLOCK-END
