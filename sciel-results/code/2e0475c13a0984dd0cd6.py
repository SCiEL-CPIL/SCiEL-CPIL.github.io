"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial terms (up to degree 3) combined with
    trigonometric terms to capture both smooth and oscillatory behavior.
    
    The basis is designed to be well-conditioned for BFGS optimization
    starting from all ones. We use a mix of:
    - Constant and linear terms
    - Quadratic and cubic polynomial terms
    - Sine/cosine terms at frequency pi and 2*pi
    - Interaction terms between x and y
    
    10 basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: cos(pi*x)
    7: cos(pi*y)
    8: sin(pi*x)*sin(pi*y)
    9: cos(pi*x)*cos(pi*y)
    """
    # Polynomial terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    
    # Trigonometric terms
    term6 = np.cos(np.pi * x)
    term7 = np.cos(np.pi * y)
    term8 = np.sin(np.pi * x) * np.sin(np.pi * y)
    term9 = np.cos(np.pi * x) * np.cos(np.pi * y)
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
