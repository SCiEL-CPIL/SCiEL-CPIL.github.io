# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of a fixed set of basis functions.
    
    Basis functions chosen to cover common symbolic regression targets:
    - Polynomials (1, x, x^2, x^3)
    - Trigonometric (sin(x), cos(x), sin(2x))
    - Hyperbolic tangent and its product with x (tanh(x), x*tanh(x))
    - Interaction term (x*sin(x))
    
    params: array of 10 coefficients.
    x: array of input points.
    """
    x = np.asarray(x, dtype=float)
    
    # Compute basis vectors
    b1 = np.ones_like(x)
    b2 = x
    b3 = x**2
    b4 = x**3
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.tanh(x)
    b8 = x * np.tanh(x)
    b9 = np.sin(2.0 * x)
    b10 = x * np.sin(x)
    
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
