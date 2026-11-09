"""Symbolic regression with 1 input(s) using a hybrid Polynomial-Fourier-Exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - Low-order polynomials (1, x, x^2, x^3) for global trends.
    - Fourier terms (sin, cos) at multiple frequencies for oscillatory behavior.
    - Exponential terms (exp(-x), exp(x)) for decay/growth.
    
    This basis is chosen to cover a wide range of common functional forms
    likely to appear in symbolic regression tasks.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the hybrid basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # Column 0: constant term (bias)
    basis[:, 0] = 1.0
    
    # Columns 1-3: polynomial terms x, x^2, x^3
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    
    # Columns 4-7: Fourier terms at two frequencies
    # Frequency 1: sin(x), cos(x)
    basis[:, 4] = np.sin(x)
    basis[:, 5] = np.cos(x)
    # Frequency 2: sin(2x), cos(2x)
    basis[:, 6] = np.sin(2 * x)
    basis[:, 7] = np.cos(2 * x)
    
    # Columns 8-9: exponential terms
    basis[:, 8] = np.exp(-x)
    basis[:, 9] = np.exp(x)
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
