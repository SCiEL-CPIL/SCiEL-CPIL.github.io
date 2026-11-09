"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # New Approach: Polynomial Regression of degree 9.
    # Parameters:
    # 0: c0 (constant term)
    # 1: c1 (x term)
    # 2: c2 (x^2 term)
    # ...
    # 9: c9 (x^9 term)
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x7 = x6 * x
    x8 = x4 * x4
    x9 = x8 * x
    
    return (params[0] + 
            params[1] * x + 
            params[2] * x2 + 
            params[3] * x3 + 
            params[4] * x4 + 
            params[5] * x5 + 
            params[6] * x6 + 
            params[7] * x7 + 
            params[8] * x8 + 
            params[9] * x9)
# EVOLVE-BLOCK-END
