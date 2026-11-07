"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis for symbolic regression on [-1,1]^2.
    Uses well-conditioned, smooth basis functions to help BFGS converge.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Basis functions (10 total), all smooth and bounded on [-1,1]
    b0 = np.ones_like(x)                          # constant
    b1 = x                                        # linear x
    b2 = y                                        # linear y
    b3 = x * x                                    # quadratic x
    b4 = y * y                                    # quadratic y
    b5 = x * y                                    # cross term
    b6 = np.sin(np.pi * x)                        # periodic in x
    b7 = np.sin(np.pi * y)                        # periodic in y
    b8 = np.cos(np.pi * (x + y))                  # diagonal oscillation
    b9 = x * y * np.sqrt(x * x + y * y + 1e-8)   # weighted cross term
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
