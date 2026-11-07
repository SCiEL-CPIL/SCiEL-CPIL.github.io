"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rich basis of functions.
    
    We use 10 parameters as coefficients for a linear combination of basis functions.
    The basis is chosen to cover common functional forms:
    - Polynomial terms: 1, x, x^2, x^3
    - Exponential: exp(x), exp(-x)
    - Logarithmic: log(x)
    - Trigonometric: sin(x), cos(x)
    - Rational: 1/x
    
    To improve conditioning for BFGS, we standardize the basis functions
    based on the range of x (assumed [0.1, 6]).
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Compute basis functions
    # We'll use 10 basis functions corresponding to the 10 params
    
    # Standardize x to improve conditioning
    # x is in [0.1, 6], so let's shift and scale
    x_shifted = (x - 0.1) / (6.0 - 0.1)  # Now in [0, 1]
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x_shifted
    b2 = x_shifted ** 2
    b3 = x_shifted ** 3
    b4 = np.exp(x_shifted)
    b5 = np.exp(-x_shifted)
    b6 = np.log(x + 1.0)  # log(x+1) to avoid log(0) issues, though x>=0.1
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = 1.0 / (x + 0.1)  # 1/(x+0.1) to avoid division by zero
    
    # Combine with parameters
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
