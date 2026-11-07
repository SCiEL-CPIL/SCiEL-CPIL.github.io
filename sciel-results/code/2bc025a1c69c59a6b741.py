"""Bootstrap portfolio: Richer basis with higher-order polynomials and interactions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a richer basis that includes:
    - Polynomial terms up to degree 3 in x and y
    - Cross terms
    - Non-polynomial terms (exp, sin, cos)
    
    This covers a wider range of potential target functions.
    """
    p = params
    
    # Polynomial terms in x
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    
    # Polynomial terms in y
    b4 = y
    b5 = y * y
    b6 = y * y * y
    
    # Cross terms
    b7 = x * y
    b8 = x * y * (x + y)  # degree 3 cross term
    
    # Non-polynomial terms
    b9 = np.sin(x) + np.sin(y)
    
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    return result
# EVOLVE-BLOCK-END
