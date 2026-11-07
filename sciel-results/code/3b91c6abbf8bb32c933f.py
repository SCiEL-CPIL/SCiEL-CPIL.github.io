"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a cubic polynomial basis in x and y.
    
    The model is:
    z = p[0] + p[1]*x + p[2]*y 
      + p[3]*x^2 + p[4]*x*y + p[5]*y^2 
      + p[6]*x^3 + p[7]*x^2*y + p[8]*x*y^2 + p[9]*y^3
    
    This utilizes all 10 parameters to span the space of degree-3 polynomials.
    """
    p = params
    
    # Ensure parameters are treated as floats to avoid integer division issues if params are passed as ints
    p = p.astype(float)
    
    # Compute basis terms
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Construct the linear combination
    # p[0]: constant
    # p[1]: x
    # p[2]: y
    # p[3]: x^2
    # p[4]: xy
    # p[5]: y^2
    # p[6]: x^3
    # p[7]: x^2*y
    # p[8]: x*y^2
    # p[9]: y^3
    
    result = (p[0] 
              + p[1] * x 
              + p[2] * y 
              + p[3] * x2 
              + p[4] * xy 
              + p[5] * y2 
              + p[6] * x3 
              + p[7] * x2y 
              + p[8] * xy2 
              + p[9] * y3)
              
    return result
# EVOLVE-BLOCK-END
