"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of a flexible basis of functions.
    The basis is designed to capture polynomial, periodic, and smooth non-linear behaviors
    for inputs in the range [-3, 3].
    
    params: array of length 10, corresponding to coefficients for the basis functions.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    # Construct the basis matrix
    # 1. Constant
    # 2. x
    # 3. x^2
    # 4. x^3
    # 5. x^4
    # 6. sin(x)
    # 7. cos(x)
    # 8. tanh(x)
    # 9. exp(-x^2)
    # 10. x * sin(x)
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        np.sin(x),
        np.cos(x),
        np.tanh(x),
        np.exp(-x ** 2),
        x * np.sin(x)
    ])
    
    # params is a 1D array of length 10
    # We perform the linear combination
    return basis @ params

# EVOLVE-BLOCK-END
