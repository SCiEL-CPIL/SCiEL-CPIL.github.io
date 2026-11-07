"""Symbolic regression with a rich, well-conditioned basis for BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _build_basis(x: np.ndarray) -> np.ndarray:
    """
    Build a basis matrix of shape (n, 10) from input x in [0.1, 6].
    
    We use a combination of:
    - Polynomial terms (up to degree 4)
    - Fourier terms (sine/cosine with scaled frequencies)
    - Scaled x to improve conditioning
    
    The basis is designed to be smooth and well-conditioned so that BFGS
    starting from all-ones can converge to a good local minimum.
    """
    n = len(x)
    
    # Scale x to [0, 1] for better conditioning
    x_min, x_max = 0.1, 6.0
    z = (x - x_min) / (x_max - x_min)  # z in [0, 1]
    
    # Scale to [0, 2*pi] for Fourier terms
    theta = 2 * np.pi * z
    
    # Build basis columns
    b0 = np.ones(n)          # constant
    b1 = x                   # linear
    b2 = x ** 2              # quadratic
    b3 = x ** 3              # cubic
    b4 = x ** 4              # quartic
    b5 = np.sin(theta)       # first harmonic sine
    b6 = np.cos(theta)       # first harmonic cosine
    b7 = np.sin(2 * theta)   # second harmonic sine
    b8 = np.cos(2 * theta)   # second harmonic cosine
    b9 = np.sin(3 * theta)   # third harmonic sine
    
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of basis functions.
    params is a 10-element array of coefficients.
    """
    basis = _build_basis(x)
    return basis @ params
# EVOLVE-BLOCK-END
