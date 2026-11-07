"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a rich basis of functions that can represent many common target functions.
    We use a Chebyshev-like polynomial basis plus trigonometric and exponential terms.
    The 10 params are used as coefficients in a linear combination.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build a diverse but stable basis for x in [-3, 3]
    # Using Chebyshev polynomials for numerical stability
    # T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x, T4=8x^4-8x^2+1, T5=16x^5-20x^3+5x
    
    # Scale x to [-1, 1] for Chebyshev stability
    xs = x / 3.0  # now in [-1, 1]
    
    # Chebyshev polynomials up to degree 5
    T0 = np.ones_like(xs)
    T1 = xs
    T2 = 2 * xs**2 - 1
    T3 = 4 * xs**3 - 3 * xs
    T4 = 8 * xs**4 - 8 * xs**2 + 1
    T5 = 16 * xs**5 - 20 * xs**3 + 5 * xs
    
    # Additional basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_x = np.exp(x)  # safe for x in [-3,3]
    exp_neg_x = np.exp(-x)
    
    basis = np.column_stack([
        T0, T1, T2, T3, T4, T5,
        sin_x, cos_x, exp_x, exp_neg_x
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
