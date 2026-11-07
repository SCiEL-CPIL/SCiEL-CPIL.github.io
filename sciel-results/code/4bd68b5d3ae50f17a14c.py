"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a mixed basis of polynomials and trigonometric functions.
    
    Basis functions (10 total):
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: sin(x)
    7: cos(y)
    8: sin(x + y)
    9: cos(x - y)
    
    This basis spans quadratic polynomials and common trigonometric forms,
    allowing BFGS to approximate a wide class of smooth functions on [-1,1]^2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Polynomial basis (degree <= 2)
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x ** 2
    b4 = y ** 2
    b5 = x * y
    
    # Trigonometric basis
    b6 = np.sin(x)
    b7 = np.cos(y)
    b8 = np.sin(x + y)
    b9 = np.cos(x - y)
    
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
