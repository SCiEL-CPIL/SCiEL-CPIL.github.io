# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    
    Strategy: Use a basis of reciprocal and interaction terms to capture rational/separable structures.
    Based on observation that reciprocal/interaction basis previously scored better than pure polynomials.
    
    Basis functions (10 total):
    0: 1/x
    1: 1/y
    2: 1/z
    3: x*y
    4: x*z
    5: y*z
    6: x*y*z
    7: 1/(x*y)
    8: 1/(x*z)
    9: 1/(y*z)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    xy = x * y
    xz = x * z
    yz = y * z
    xyz = x * y * z
    
    inv_xy = 1.0 / (x * y)
    inv_xz = 1.0 / (x * z)
    inv_yz = 1.0 / (y * z)
    
    # Build prediction
    pred = (
        p0 * inv_x +
        p1 * inv_y +
        p2 * inv_z +
        p3 * xy +
        p4 * xz +
        p5 * yz +
        p6 * xyz +
        p7 * inv_xy +
        p8 * inv_xz +
        p9 * inv_yz
    )
    
    return pred
# EVOLVE-BLOCK-END
