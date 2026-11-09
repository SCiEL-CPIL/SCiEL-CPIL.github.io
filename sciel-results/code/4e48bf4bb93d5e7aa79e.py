"""Symbolic regression with a well-conditioned, diverse basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]

    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x / 2.0

    # Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    b4 = 8.0 * t**4 - 8.0 * t**2 + 1.0

    # Trigonometric functions (period 2 and 1)
    b5 = np.sin(np.pi * x)
    b6 = np.cos(np.pi * x)
    b7 = np.sin(2.0 * np.pi * x)

    # Gaussian functions (localized)
    # Centered at 0
    b8 = np.exp(-x**2)
    # Symmetric centered at +/- 1
    b9 = np.exp(-(x - 1.0)**2) + np.exp(-(x + 1.0)**2)

    # Combine basis functions with parameters
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
