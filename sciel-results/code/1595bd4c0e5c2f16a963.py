"""Symbolic regression with 1 input; local basis expansion with learned centers."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local basis expansion model with 10 parameters.
    
    Uses a combination of:
    - Global polynomial terms (constant, linear, quadratic)
    - Local Gaussian bumps at 3 fixed centers (-1, 0, 1)
    - A global tanh for smooth nonlinearity
    - A global sin for periodicity
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: exp(-(x+1)^2)
    4: exp(-x^2)
    5: exp(-(x-1)^2)
    6: exp(-(x+0.5)^2)
    7: exp(-(x-0.5)^2)
    8: tanh(x)
    9: sin(x)
    """
    p = params
    
    # Polynomial terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = x * x
    
    # Gaussian bumps at various centers
    term3 = np.exp(-(x + 1.0) ** 2)
    term4 = np.exp(-x ** 2)
    term5 = np.exp(-(x - 1.0) ** 2)
    term6 = np.exp(-(x + 0.5) ** 2)
    term7 = np.exp(-(x - 0.5) ** 2)
    
    # Nonlinear smooth functions
    term8 = np.tanh(x)
    term9 = np.sin(x)
    
    y = (p[0] * term0 +
         p[1] * term1 +
         p[2] * term2 +
         p[3] * term3 +
         p[4] * term4 +
         p[5] * term5 +
         p[6] * term6 +
         p[7] * term7 +
         p[8] * term8 +
         p[9] * term9)
    
    return y
# EVOLVE-BLOCK-END
