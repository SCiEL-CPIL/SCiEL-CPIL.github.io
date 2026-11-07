# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters for symbolic regression.
    
    We use a rich basis of monomials, trigonometric, and rational terms
    to span a large function space. The evaluator fits params via BFGS.
    
    Basis functions (10 total):
    0: x
    1: y
    2: z
    3: x*y
    4: x*z
    5: y*z
    6: x*y*z
    7: x^2
    8: y^2
    9: z^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    pred = (
        p0 * x +
        p1 * y +
        p2 * z +
        p3 * x * y +
        p4 * x * z +
        p5 * y * z +
        p6 * x * y * z +
        p7 * x * x +
        p8 * y * y +
        p9 * z * z
    )
    
    return pred
# EVOLVE-BLOCK-END
