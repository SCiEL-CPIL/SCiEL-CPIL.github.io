"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    We use a basis that includes polynomial terms up to degree 3
    to allow fitting a wide variety of smooth functions.
    
    Basis (10 terms):
    0: 1
    1: x
    2: y
    3: z
    4: x*y
    5: x*z
    6: y*z
    7: x^2
    8: y^2
    9: z^2
    """
    p = params
    return (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * y +
        p[3] * z +
        p[4] * x * y +
        p[5] * x * z +
        p[6] * y * z +
        p[7] * x**2 +
        p[8] * y**2 +
        p[9] * z**2
    )
# EVOLVE-BLOCK-END
