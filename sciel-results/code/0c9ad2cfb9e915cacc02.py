"""Symbolic regression with 1 input using a diverse basis of candidate functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a linear combination of 10 diverse basis functions that can capture
    a wide range of functional forms: polynomial, exponential, logarithmic,
    trigonometric, and rational terms.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Safe x for log and division
    x_safe = np.clip(x, 0.01, 10.0)
    
    # Build basis matrix with 10 diverse features
    basis = np.empty((len(x), 10), dtype=np.float64)
    
    # Polynomial terms (degrees 0-3)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    
    # Exponential terms
    basis[:, 4] = np.exp(-x)
    basis[:, 5] = np.exp(x)
    
    # Logarithmic term
    basis[:, 6] = np.log(x_safe)
    
    # Trigonometric terms
    basis[:, 7] = np.sin(x)
    basis[:, 8] = np.cos(x)
    
    # Rational term
    basis[:, 9] = 1.0 / x_safe
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
