"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bivariate cubic polynomial basis.
    The model is:
    p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 + 
    p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    
    This basis has 10 terms, matching the 10 parameters.
    It is linear in parameters, ensuring the optimization problem is convex
    and BFGS will converge to the global minimum for this basis.
    """
    # Calculate powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Construct prediction
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y
            + params[3] * x2
            + params[4] * x * y
            + params[5] * y2
            + params[6] * x3
            + params[7] * x2 * y
            + params[8] * x * y2
            + params[9] * y3)
    
    return pred
# EVOLVE-BLOCK-END
