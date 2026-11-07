"""Symbolic regression with a rich, stable linear basis for BFGS-compatible optimization."""
import numpy as np
import time

# EVOLVE-BLOCK-START


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters as coefficients of a well-conditioned
    linear basis. The basis spans polynomials, trigonometric functions, and
    hyperbolic functions to cover common target function families.
    
    Basis functions:
    - p[0] * 1
    - p[1] * x
    - p[2] * x^2
    - p[3] * x^3
    - p[4] * x^4
    - p[5] * sin(x)
    - p[6] * cos(x)
    - p[7] * sin(2*x)
    - p[8] * cos(2*x)
    - p[9] * tanh(x)
    
    This is a linear model in params, making the loss quadratic and BFGS
    optimization straightforward and robust.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Build basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    basis = np.column_stack([
        np.ones_like(x),       # p[0]
        x,                      # p[1]
        x2,                     # p[2]
        x3,                     # p[3]
        x4,                     # p[4]
        np.sin(x),             # p[5]
        np.cos(x),             # p[6]
        np.sin(2.0 * x),       # p[7]
        np.cos(2.0 * x),       # p[8]
        np.tanh(x),            # p[9]
    ])
    
    # Linear combination
    result = basis @ p
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result


# EVOLVE-BLOCK-END
