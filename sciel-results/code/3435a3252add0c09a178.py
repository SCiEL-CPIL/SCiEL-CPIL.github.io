"""Symbolic regression with flexible basis expansion."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a flexible basis expansion.
    
    We use a combination of polynomial terms up to degree 5, plus logarithmic,
    exponential, and trigonometric terms to capture a wide variety of functional forms.
    
    params (length 10) are linear coefficients for the basis functions:
        1, x, x^2, x^3, x^4, x^5, log(x), exp(-x), sin(x), cos(x)
    
    This linear-in-parameters model allows BFGS to efficiently find the best
    linear combination. Since the model is linear in params, BFGS should converge
    quickly to the global optimum for the given basis.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Build basis functions
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        x**5,
        np.log(x + 1e-10),  # small epsilon to avoid log(0)
        np.exp(-x),
        np.sin(x),
        np.cos(x)
    ])
    
    # Linear combination of basis functions with params
    return basis @ params
# EVOLVE-BLOCK-END
