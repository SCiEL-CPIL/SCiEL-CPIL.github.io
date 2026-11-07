# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Pure bivariate polynomial expansion (Taylor series) in strain and temp.
    Uses 10 coefficients for a degree-3 polynomial in strain and degree-2 in temp.
    
    sigma = c0 + c1*s + c2*s^2 + c3*s^3 + c4*t + c5*s*t + c6*s^2*t + c7*t^2 + c8*s*t^2 + c9*s^2*t^2
    
    params:
    [0]: c0 (constant)
    [1]: c1 (strain linear)
    [2]: c2 (strain quadratic)
    [3]: c3 (strain cubic)
    [4]: c4 (temp linear)
    [5]: c5 (strain*temp)
    [6]: c6 (strain^2*temp)
    [7]: c7 (temp^2)
    [8]: c8 (strain*temp^2)
    [9]: c9 (strain^2*temp^2)
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Unpack parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Precompute powers
    s = strain
    t = temp
    s2 = s * s
    s3 = s2 * s
    t2 = t * t
    
    # Bivariate polynomial expansion
    stress = (c0 
              + c1 * s 
              + c2 * s2 
              + c3 * s3 
              + c4 * t 
              + c5 * s * t 
              + c6 * s2 * t 
              + c7 * t2 
              + c8 * s * t2 
              + c9 * s2 * t2)
    
    return stress
# EVOLVE-BLOCK-END
