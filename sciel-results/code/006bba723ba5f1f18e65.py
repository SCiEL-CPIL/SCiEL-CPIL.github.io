"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs using a rich basis of 10 features.
    
    The basis includes:
    1. x
    2. y
    3. x^2
    4. y^2
    5. x*y
    6. x^3
    7. y^3
    8. x^2*y
    9. x*y^2
    10. x*y*(x^2 + y^2)  [captures radial symmetry]
    
    This is a linear-in-parameters model, so BFGS will converge robustly.
    The basis is designed to capture common functional forms:
    - Polynomials up to degree 3
    - Radial symmetry (x^2 + y^2 terms)
    - Mixed terms
    """
    # Compute basis features
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # 10 basis functions
    f1 = x
    f2 = y
    f3 = x2
    f4 = y2
    f5 = xy
    f6 = x * x2      # x^3
    f7 = y * y2      # y^3
    f8 = x2 * y      # x^2 * y
    f9 = x * y2      # x * y^2
    f10 = xy * (x2 + y2)  # x*y*(x^2 + y^2)
    
    # Linear combination
    pred = (params[0] * f1 + params[1] * f2 + params[2] * f3 + 
            params[3] * f4 + params[4] * f5 + params[5] * f6 + 
            params[6] * f7 + params[7] * f8 + params[8] * f9 + 
            params[9] * f10)
    
    return pred
# EVOLVE-BLOCK-END
