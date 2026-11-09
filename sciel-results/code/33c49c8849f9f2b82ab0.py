"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a rich basis of 10 features.
    
    The basis includes:
    - Constant
    - Scaled polynomial terms (u to u^5) where u is x scaled to [-1, 1]
    - Exponential terms (exp(u), exp(-u))
    - Trigonometric terms (sin(2u), cos(2u))
    
    x is scaled from [0.1, 6] to u in approximately [-1, 1] for numerical stability.
    """
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    u = (x - 3.05) / 2.95
    
    # Construct basis terms
    b0 = np.ones_like(x)
    b1 = u
    b2 = u * u
    b3 = b2 * u
    b4 = b3 * u
    b5 = b4 * u
    b6 = np.exp(u)
    b7 = np.exp(-u)
    b8 = np.sin(2.0 * u)
    b9 = np.cos(2.0 * u)
    
    # Combine with parameters
    return (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
# EVOLVE-BLOCK-END
