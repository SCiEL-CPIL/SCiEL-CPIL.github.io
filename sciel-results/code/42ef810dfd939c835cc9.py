import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of a fixed set of basis functions.
    
    Basis functions chosen to cover:
    - Polynomials (1, x, x^2, x^3, x^4)
    - Trigonometric (sin(x), cos(x), sin(2x))
    - Hyperbolic tangent (tanh(x)) for saturation
    - Scaled hyperbolic tangent (x * tanh(x))
    
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
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.sin(2.0 * x)
    b9 = np.tanh(x)
    b10 = x * np.tanh(x)
    
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
