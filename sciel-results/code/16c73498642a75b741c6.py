"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial basis up to degree 3 (10 terms), linear in parameters.
    Basis:
      p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 +
      p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    This is a complete bivariate cubic polynomial, providing a rich but stable
    representation. Since the model is linear in parameters, BFGS converges to
    the global minimum quickly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements; pad with zeros if needed
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p[:10]
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    pred = (p0 
            + p1 * x 
            + p2 * y 
            + p3 * x2 
            + p4 * x * y 
            + p5 * y2 
            + p6 * x3 
            + p7 * x2 * y 
            + p8 * x * y2 
            + p9 * y3)
    
    return pred
# EVOLVE-BLOCK-END
