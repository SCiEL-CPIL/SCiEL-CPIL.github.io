"""Symbolic regression with 1 input: adaptive basis + safe evaluation."""
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 140.0  # 140s safety margin before 150s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    
    We use a rich set of basis functions to span a wide variety of smooth
    functions on [-3, 3]:
    - Polynomial terms (up to degree 5)
    - Trigonometric terms (sin, cos at multiple frequencies)
    - Exponential terms (exp, exp(-x))
    - Hyperbolic terms (tanh, sech-like)
    
    The model is linear in params, so BFGS converges to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Build basis matrix (n_samples, 10)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Polynomial basis (degree 0-4)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    
    # Trigonometric basis
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.sin(2.0 * x)
    basis[:, 8] = np.cos(2.0 * x)
    
    # Hyperbolic / exponential basis
    # Use tanh for bounded smooth transition
    basis[:, 9] = np.tanh(x)
    
    # Compute prediction
    pred = basis @ params
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
