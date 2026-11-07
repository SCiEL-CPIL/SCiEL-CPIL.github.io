# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Fourier series expansion using sine and cosine terms of various
    frequencies to target periodic/oscillatory target functions. The 10 parameters
    are coefficients for:
      1. sin(x)
      2. cos(x)
      3. sin(y)
      4. cos(y)
      5. sin(x+y)
      6. cos(x+y)
      7. sin(x-y)
      8. cos(x-y)
      9. sin(2*x)
      10. cos(2*y)
    
    This leverages the orthogonality of trigonometric functions on [-1,1]
    to allow BFGS to efficiently fit periodic components.
    """
    p = params
    
    # Compute Fourier basis terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    sin_xy_plus = np.sin(x + y)
    cos_xy_plus = np.cos(x + y)
    sin_xy_minus = np.sin(x - y)
    cos_xy_minus = np.cos(x - y)
    sin_2x = np.sin(2 * x)
    cos_2y = np.cos(2 * y)
    
    # Linear combination with learned coefficients
    out = (p[0] * sin_x +
           p[1] * cos_x +
           p[2] * sin_y +
           p[3] * cos_y +
           p[4] * sin_xy_plus +
           p[5] * cos_xy_plus +
           p[6] * sin_xy_minus +
           p[7] * cos_xy_minus +
           p[8] * sin_2x +
           p[9] * cos_2y)
    
    return out
# EVOLVE-BLOCK-END
