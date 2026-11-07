"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis using Chebyshev polynomials for low-order structure
    and Fourier terms for potential oscillatory behavior.
    
    Basis functions (10 total):
      0: 1
      1: x
      2: y
      3: 2x^2 - 1 (T2(x))
      4: 2y^2 - 1 (T2(y))
      5: xy
      6: cos(2*pi*x)
      7: cos(2*pi*y)
      8: sin(2*pi*x)
      9: sin(2*pi*y)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = 2 * x**2 - 1
    b4 = 2 * y**2 - 1
    b5 = x * y
    
    # Fourier terms with frequency 2*pi (period 1)
    freq = 2 * np.pi
    b6 = np.cos(freq * x)
    b7 = np.cos(freq * y)
    b8 = np.sin(freq * x)
    b9 = np.sin(freq * y)
    
    # Linear combination
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return pred
# EVOLVE-BLOCK-END
