"""Symbolic regression with a well-conditioned linear basis for BFGS-compatible optimization."""
import numpy as np

# EVOLVE-BLOCK-START


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of 10 fixed basis functions.
    
    The basis is designed to be well-conditioned and span a rich function space
    over [-2, 2], making the optimization problem convex (linear in params)
    so BFGS can solve it robustly and globally.
    
    Basis functions:
    - p[0]: constant (1)
    - p[1]: x
    - p[2]: x^2
    - p[3]: x^3
    - p[4]: x^4
    - p[5]: sin(x)
    - p[6]: cos(x)
    - p[7]: sin(2x)
    - p[8]: cos(2x)
    - p[9]: tanh(x)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Build basis matrix (n_points x 10)
    # Using fixed basis functions for convex optimization
    basis = np.column_stack([
        np.ones_like(x),       # constant
        x,                      # linear
        x ** 2,                 # quadratic
        x ** 3,                 # cubic
        x ** 4,                 # quartic
        np.sin(x),              # sin
        np.cos(x),              # cos
        np.sin(2.0 * x),        # sin(2x)
        np.cos(2.0 * x),        # cos(2x)
        np.tanh(x),             # tanh
    ])
    
    # Linear combination
    result = basis @ p
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result


# EVOLVE-BLOCK-END
