"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of a fixed set of basis functions.
    
    Basis functions chosen to cover:
    - Polynomials (1, x, x^2, x^3, x^4, x^5)
    - Trigonometric (sin(x), cos(x), sin(2x), cos(2x))
    - Hyperbolic tangent (tanh(x)) for saturation
    
    params: array of 10 coefficients.
    x: array of input points.
    """
    # Ensure x is float for calculation
    x = np.asarray(x, dtype=float)
    
    # Compute basis vectors
    b1 = np.ones_like(x)
    b2 = x
    b3 = x**2
    b4 = x**3
    b5 = x**4
    b6 = x**5
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.sin(2.0 * x)
    b10 = np.tanh(x)
    
    # Linear combination
    result = (params[0] * b1 + 
              params[1] * b2 + 
              params[2] * b3 + 
              params[3] * b4 + 
              params[4] * b5 + 
              params[5] * b6 + 
              params[6] * b7 + 
              params[7] * b8 + 
              params[8] * b9 + 
              params[9] * b10)
              
    return result
# EVOLVE-BLOCK-END
