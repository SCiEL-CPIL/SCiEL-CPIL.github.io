"""Symbolic regression: recover hidden function from 1 input with 10 free parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure we have at least 10 params
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), constant_values=1.0)
    p = p[:10]
    
    # Build a diverse set of basis functions that can capture many functional forms
    # on [-2, 2] with only 10 parameters.
    
    # Polynomial-like terms
    b0 = np.ones_like(x)              # constant
    b1 = x                            # linear
    b2 = x ** 2                       # quadratic
    b3 = x ** 3                       # cubic
    b4 = x ** 4                       # quartic
    
    # Trigonometric terms (capture periodic/oscillatory behavior)
    b5 = np.sin(x)                    # sine
    b6 = np.cos(x)                    # cosine
    b7 = np.sin(2.0 * x)              # double-frequency sine
    
    # Nonlinear smooth terms
    b8 = np.tanh(x)                   # hyperbolic tangent (smooth step-like)
    b9 = np.exp(-x ** 2)              # Gaussian bump (centered)
    
    # Combine with parameters
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    # Guard against non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
