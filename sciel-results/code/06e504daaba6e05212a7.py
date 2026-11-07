"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a hybrid polynomial-rational basis.
    Terms: 1, x, y, z, x^2, y^2, z^2, x/y, y/z, z/x
    
    This basis captures linear and quadratic behaviors as well as 
    cyclic rational dependencies, which are often present in symbolic 
    regression targets involving division or ratios.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Compute basis elements
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    # Ratio terms (safe as x,y,z in [0.5, 2])
    r_xy = x / y
    r_yz = y / z
    r_zx = z / x
    
    # Linear combination
    return c0 + c1 * x + c2 * y + c3 * z + c4 * x2 + c5 * y2 + c6 * z2 + c7 * r_xy + c8 * r_yz + c9 * r_zx
# EVOLVE-BLOCK-END
