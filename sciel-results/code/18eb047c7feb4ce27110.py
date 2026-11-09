"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with a richer basis to recover the hidden function.
    Inputs x, y, z are in [0.5, 2].
    
    We use 10 params, so we select the most useful 10 basis functions.
    The evaluator fits params via BFGS from all ones, minimizing training MSE.
    
    Strategy: Use individual terms rather than combined ones to give BFGS
    more flexibility. Include polynomial, reciprocal, and trig terms.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis functions (10 terms) - individual terms for better flexibility
    
    # f0: constant
    f0 = np.ones_like(x)
    
    # f1: x
    f1 = x
    
    # f2: y
    f2 = y
    
    # f3: z
    f3 = z
    
    # f4: xy
    f4 = x * y
    
    # f5: xz
    f5 = x * z
    
    # f6: yz
    f6 = y * z
    
    # f7: 1/x (reciprocal)
    f7 = 1.0 / x
    
    # f8: 1/y (reciprocal)
    f8 = 1.0 / y
    
    # f9: 1/z (reciprocal)
    f9 = 1.0 / z
    
    result = (params[0] * f0 +
              params[1] * f1 +
              params[2] * f2 +
              params[3] * f3 +
              params[4] * f4 +
              params[5] * f5 +
              params[6] * f6 +
              params[7] * f7 +
              params[8] * f8 +
              params[9] * f9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
