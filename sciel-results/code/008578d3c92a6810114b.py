"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 basis functions.
    
    Basis functions chosen to span a wide variety of common mathematical forms:
    - Polynomials (degree 0-4)
    - Trigonometric (sin(x), cos(x))
    - Gaussian decay (exp(-x^2))
    - Saturating non-linearities (tanh(x))
    - Interaction term (x * exp(-x^2))
    
    This linear model is convex with respect to parameters, allowing BFGS to find
    the global minimum efficiently.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Pre-compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-x2)
    b8 = np.tanh(x)
    b9 = x * np.exp(-x2)
    
    # Linear combination
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
