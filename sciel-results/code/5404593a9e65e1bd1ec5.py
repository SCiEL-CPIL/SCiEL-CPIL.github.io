"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters for symbolic regression.
    
    We use a basis of constants, linear terms, reciprocals, and rational 
    interactions. This is designed to capture functions with rational structure 
    (e.g., 1/x, x/y) which are common in symbolic regression tasks.
    
    Basis functions (10 total):
    0: 1
    1: x
    2: y
    3: z
    4: 1/x
    5: 1/y
    6: 1/z
    7: x/y
    8: y/z
    9: z/x
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # Ensure inputs are float for safe division
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Build prediction
    # p0: constant
    # p1: x
    # p2: y
    # p3: z
    # p4: 1/x
    # p5: 1/y
    # p6: 1/z
    # p7: x/y
    # p8: y/z
    # p9: z/x
    
    pred = (
        p0 * 1.0 +
        p1 * x +
        p2 * y +
        p3 * z +
        p4 * inv_x +
        p5 * inv_y +
        p6 * inv_z +
        p7 * (x * inv_y) +
        p8 * (y * inv_z) +
        p9 * (z * inv_x)
    )
    
    return pred
# EVOLVE-BLOCK-END
