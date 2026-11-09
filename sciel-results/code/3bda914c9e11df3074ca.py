"""Symbolic regression with a Fourier-series basis (up to 5th harmonic)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model using a Fourier series basis up to the 5th harmonic.
    
    Basis functions (10 total, matching the 10 params):
      0: 1
      1: cos(x)
      2: sin(x)
      3: cos(2x)
      4: sin(2x)
      5: cos(3x)
      6: sin(3x)
      7: cos(4x)
      8: sin(4x)
      9: cos(5x)
    
    The evaluator fits params via BFGS starting from all ones.
    """
    # Ensure x is a 1-D array for safety
    x = np.asarray(x, dtype=np.float64)
    
    # Extract the 10 parameters
    p = params  # length-10 array
    
    # Compute all trigonometric basis functions
    c1 = np.cos(x)
    s1 = np.sin(x)
    c2 = np.cos(2.0 * x)
    s2 = np.sin(2.0 * x)
    c3 = np.cos(3.0 * x)
    s3 = np.sin(3.0 * x)
    c4 = np.cos(4.0 * x)
    s4 = np.sin(4.0 * x)
    c5 = np.cos(5.0 * x)
    
    # Linear combination: p[0]*1 + p[1]*cos(x) + p[2]*sin(x) + ...
    result = (
        p[0]
        + p[1] * c1
        + p[2] * s1
        + p[3] * c2
        + p[4] * s2
        + p[5] * c3
        + p[6] * s3
        + p[7] * c4
        + p[8] * s4
        + p[9] * c5
    )
    
    return result
# EVOLVE-BLOCK-END
