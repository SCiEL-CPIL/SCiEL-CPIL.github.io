"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a mixture of nonlinear basis functions.
    params: array of length 10.
    
    This approach uses a diverse set of basis functions to capture various
    functional forms (polynomial, periodic, exponential, sigmoidal, rational).
    Input x is scaled to z = x/3 to keep values in [-1, 1] for stability.
    
    Basis functions:
    1.  1
    2.  z
    3.  z^2
    4.  z^3
    5.  sin(x)
    6.  cos(x)
    7.  exp(-|x|)
    8.  tanh(x)
    9.  exp(-x^2/2)
    10. 1 / (1 + |x|)
    """
    z = x / 3.0
    
    # Compute basis functions
    b1 = np.ones_like(x)
    b2 = z
    b3 = z ** 2
    b4 = z ** 3
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-np.abs(x))
    b8 = np.tanh(x)
    b9 = np.exp(-0.5 * x ** 2)
    b10 = 1.0 / (1.0 + np.abs(x))
    
    # Stack into a matrix: shape (n_points, 10)
    B = np.column_stack((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10))
    
    # Prediction: B @ params
    pred = B @ params
    
    return pred
# EVOLVE-BLOCK-END
