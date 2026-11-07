"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract 10 parameters
    p = params
    
    # Polynomial basis functions up to degree 2 for 2 inputs
    # Features: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    # This gives us a flexible polynomial model with 10 parameters
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    x2y = x2 * y
    xy2 = x * y2
    y3 = y2 * y
    
    # Linear combination of basis functions
    # params[0]: constant
    # params[1]: x
    # params[2]: y
    # params[3]: x^2
    # params[4]: xy
    # params[5]: y^2
    # params[6]: x^3
    # params[7]: x^2y
    # params[8]: xy^2
    # params[9]: y^3
    
    result = (p[0] + 
              p[1] * x + 
              p[2] * y + 
              p[3] * x2 + 
              p[4] * xy + 
              p[5] * y2 + 
              p[6] * x3 + 
              p[7] * x2y + 
              p[8] * xy2 + 
              p[9] * y3)
    
    return result
# EVOLVE-BLOCK-END
