"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a complete polynomial basis of degree 3.
    This approach captures smooth, non-periodic structures and is a universal
    approximator on bounded domains. The 10 parameters correspond to the
    coefficients of the 10 monomials in the complete polynomial of degree 3
    in two variables.
    """
    # Basis functions: Complete polynomial of degree 3
    # Terms: x^3, x^2*y, x*y^2, y^3, x^2, x*y, y^2, x, y, 1
    
    b0 = x**3
    b1 = x**2 * y
    b2 = x * y**2
    b3 = y**3
    b4 = x**2
    b5 = x * y
    b6 = y**2
    b7 = x
    b8 = y
    b9 = np.ones_like(x)
    
    # Combine with parameters
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
