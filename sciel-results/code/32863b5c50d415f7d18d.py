"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function with 10 parameters to approximate hidden function f(x).
    
    Basis functions chosen to span common function families:
    - Polynomials (degree 0-4)
    - Trigonometric (sin, cos, sin(2x), cos(2x))
    - Gaussian-like (exp(-x^2))
    - Rational-like (x/(1+x^2))
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p = params
    
    # Build basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x**2
    # 4. Cubic
    b3 = x**3
    # 5. Quartic
    b4 = x**4
    # 6. sin(x)
    b5 = np.sin(x)
    # 7. cos(x)
    b6 = np.cos(x)
    # 8. exp(-x^2) - Gaussian
    b7 = np.exp(-x**2)
    # 9. x/(1+x^2) - rational
    b8 = x / (1.0 + x**2)
    # 10. tanh(x) - sigmoid-like
    b9 = np.tanh(x)
    
    # Combine linearly
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
