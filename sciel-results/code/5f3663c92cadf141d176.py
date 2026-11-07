"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial basis with cross-terms.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: x*y
    8: y*z
    9: z*x
    
    This representation captures smooth non-linear dependencies and
    pairwise interactions, which is often more robust than trigonometric
    bases for functions defined on bounded intervals [0.5, 2].
    """
    p = params.flatten()
    
    # Ensure we have at least 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a4 = p[4]
    a5 = p[5]
    a6 = p[6]
    a7 = p[7]
    a8 = p[8]
    a9 = p[9]
    
    # Build the expression
    result = a0
    result += a1 * x
    result += a2 * y
    result += a3 * z
    result += a4 * x**2
    result += a5 * y**2
    result += a6 * z**2
    result += a7 * x * y
    result += a8 * y * z
    result += a9 * z * x
    
    return result
# EVOLVE-BLOCK-END
