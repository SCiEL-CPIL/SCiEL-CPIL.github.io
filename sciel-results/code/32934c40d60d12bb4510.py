import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a cubic polynomial basis.
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
             + p6*x^3 + p7*y^3 + p8*x^2*y + p9*x*y^2
    
    This expands the expressiveness of the linear baseline (which used p0, p1, p2 
    effectively as constant, x, y) to include quadratic and cubic interactions.
    """
    # Calculate powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2

    # Construct prediction
    # Order: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * x2 
            + params[4] * y2 
            + params[5] * xy 
            + params[6] * x3 
            + params[7] * y3 
            + params[8] * x2y 
            + params[9] * xy2)
            
    return pred
