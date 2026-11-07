import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial basis expansion model.
    This is a linear model in the parameters, which is robust for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Basis functions
    b0 = 1.0
    b1 = eps
    b2 = eps**2
    b3 = eps**3
    b4 = eps**4
    
    b5 = T
    b6 = eps * T
    b7 = eps**2 * T
    b8 = eps**3 * T
    b9 = eps**4 * T
    
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
              
    return result
