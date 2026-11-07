"""Symbolic regression with a rich, stable basis and BFGS-compatible evaluation."""
import numpy as np
import time

# EVOLVE-BLOCK-START


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters in a diverse, numerically stable basis.
    
    The basis is designed to span common function families:
    - Polynomials up to degree 4
    - Trigonometric functions (sin, cos) with multiple frequencies
    - Hyperbolic functions (tanh, sech-like)
    - Exponential decay/growth
    - Gaussian
    - Absolute value and signed power terms
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Build basis functions
    b0 = np.ones_like(x)                    # constant
    b1 = x                                  # linear
    b2 = x ** 2                             # quadratic
    b3 = x ** 3                             # cubic
    b4 = x ** 4                             # quartic
    
    b5 = np.sin(x)                          # sin(x)
    b6 = np.cos(x)                          # cos(x)
    b7 = np.sin(2.0 * x)                    # sin(2x)
    b8 = np.tanh(x)                         # tanh(x)
    b9 = np.exp(-x ** 2)                    # Gaussian
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)


# EVOLVE-BLOCK-END
