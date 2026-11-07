# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Use a rich basis of 10 features that covers:
    - Constant
    - Linear terms: x, y
    - Quadratic terms: x^2, y^2, x*y
    - Cubic terms: x^3, y^3
    - Trigonometric: sin(x), sin(y)
    
    The evaluator fits 10 parameters via BFGS starting from all ones.
    This basis is designed to capture a wide variety of smooth functions
    on the unit square, including polynomials up to degree 3 and some
    periodic behavior.
    """
    # Ensure inputs are properly shaped
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure we have exactly 10 parameters
    if p.shape[0] != 10:
        # Pad or truncate to 10
        p_new = np.ones(10, dtype=np.float64)
        n = min(p.shape[0], 10)
        p_new[:n] = p[:n]
        p = p_new
    
    # Build the 10 basis features
    term0 = 1.0                          # constant
    term1 = x                            # x
    term2 = y                            # y
    term3 = x * x                        # x^2
    term4 = y * y                        # y^2
    term5 = x * y                        # x*y
    term6 = x * x * x                    # x^3
    term7 = y * y * y                    # y^3
    term8 = np.sin(x)                    # sin(x)
    term9 = np.sin(y)                    # sin(y)
    
    # Linear combination of basis functions with fitted parameters
    result = (p[0] * term0 + p[1] * term1 + p[2] * term2 +
              p[3] * term3 + p[4] * term4 + p[5] * term5 +
              p[6] * term6 + p[7] * term7 + p[8] * term8 + p[9] * term9)
    
    return result
# EVOLVE-BLOCK-END
