"""Symbolic regression with 3 inputs - rich basis approach."""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a rich basis of functions of (x, y, z) to approximate the hidden function.
    Basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    This gives a full quadratic polynomial in 3 variables, which is linear in the 10 params.
    """
    p = params.flatten()
    
    # Build the quadratic basis
    # p[0]: constant
    # p[1]: x
    # p[2]: y
    # p[3]: z
    # p[4]: x^2
    # p[5]: y^2
    # p[6]: z^2
    # p[7]: x*y
    # p[8]: x*z
    # p[9]: y*z
    
    result = (p[0] 
              + p[1] * x 
              + p[2] * y 
              + p[3] * z 
              + p[4] * x * x 
              + p[5] * y * y 
              + p[6] * z * z 
              + p[7] * x * y 
              + p[8] * x * z 
              + p[9] * y * z)
    
    return result
