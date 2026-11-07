"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal basis model: combines polynomial, trigonometric, and exponential
    terms to approximate a wide class of target functions.
    
    params layout (10 values):
      0: constant
      1: x
      2: y
      3: z
      4: sin(x)
      5: sin(y)
      6: sin(z)
      7: exp(-x)
      8: exp(-y)
      9: exp(-z)
    """
    p = params
    return (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * z
            + p[4] * np.sin(x)
            + p[5] * np.sin(y)
            + p[6] * np.sin(z)
            + p[7] * np.exp(-x)
            + p[8] * np.exp(-y)
            + p[9] * np.exp(-z))
# EVOLVE-BLOCK-END
