"""Symbolic regression with 3 inputs - trigonometric harmonic basis approach."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a linear combination of 10 trigonometric basis functions.
    
    Basis functions:
      0: 1 (constant)
      1: sin(x)
      2: cos(x)
      3: sin(y)
      4: cos(y)
      5: sin(z)
      6: cos(z)
      7: sin(x + y)
      8: sin(y + z)
      9: sin(z + x)
    
    params: array of 10 weights.
    """
    p = params.flatten()
    
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = np.sin(x)
    b2 = np.cos(x)
    b3 = np.sin(y)
    b4 = np.cos(y)
    b5 = np.sin(z)
    b6 = np.cos(z)
    b7 = np.sin(x + y)
    b8 = np.sin(y + z)
    b9 = np.sin(z + x)
    
    # Linear combination
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
