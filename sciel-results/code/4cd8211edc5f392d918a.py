"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich 10-term basis for symbolic regression.
    
    The 10 parameters are coefficients for a linear combination of 10 basis functions:
    1. 1 (constant)
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. exp(x)
    7. exp(-x)
    8. log(x + 1)
    9. sin(x)
    10. cos(x)
    
    This provides BFGS with a broad search space to fit diverse hidden functions.
    The basis is well-scaled and diverse, improving conditioning over the 2-param seed.
    """
    # Compute basis functions
    basis = np.column_stack([
        np.ones_like(x),       # constant
        x,                      # linear
        x ** 2,                 # quadratic
        x ** 3,                 # cubic
        x ** 4,                 # quartic
        np.exp(x),              # exponential growth
        np.exp(-x),             # exponential decay
        np.log(x + 1.0),        # logarithmic
        np.sin(x),              # sinusoidal
        np.cos(x)               # cosinusoidal
    ])
    
    # Linear combination of basis functions
    return basis @ params
# EVOLVE-BLOCK-END
