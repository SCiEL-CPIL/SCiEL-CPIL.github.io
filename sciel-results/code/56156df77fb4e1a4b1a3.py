"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis symbolic regression with 10 parameters.
    
    Basis functions (smooth, differentiable for BFGS):
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: exp(-x^2)
    8: 1/(1+x^2)
    9: x*exp(-x^2)
    """
    p = params
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Polynomial terms
    f0 = 1.0
    f1 = x
    f2 = x2
    f3 = x3
    f4 = x4
    
    # Trigonometric
    f5 = np.sin(x)
    f6 = np.cos(x)
    
    # Gaussian bump
    f7 = np.exp(-x2)
    
    # Lorentzian
    f8 = 1.0 / (1.0 + x2)
    
    # Odd Gaussian bump
    f9 = x * np.exp(-x2)
    
    return (p[0] * f0 + p[1] * f1 + p[2] * f2 + p[3] * f3 + p[4] * f4 +
            p[5] * f5 + p[6] * f6 + p[7] * f7 + p[8] * f8 + p[9] * f9)
# EVOLVE-BLOCK-END
