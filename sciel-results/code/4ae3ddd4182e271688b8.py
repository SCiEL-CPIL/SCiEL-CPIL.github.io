"""Symbolic regression with 1 input: rich basis + safe evaluation."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    
    Basis functions chosen to span a wide variety of smooth functions on [-3, 3]:
    - Polynomial terms (up to degree 4) for general shape
    - Trigonometric terms for periodic/oscillatory behavior
    - Exponential terms for growth/decay
    - Hyperbolic terms for saturation/transition
    
    The model is linear in params, so BFGS will converge to the global minimum
    of the MSE objective (convex quadratic in params).
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Build basis matrix (n_samples, 10)
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Polynomial basis
    basis[:, 0] = 1.0          # constant
    basis[:, 1] = x            # linear
    basis[:, 2] = x ** 2       # quadratic
    basis[:, 3] = x ** 3       # cubic
    basis[:, 4] = x ** 4       # quartic
    
    # Trigonometric basis
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.sin(2.0 * x)
    basis[:, 8] = np.cos(2.0 * x)
    
    # Exponential/hyperbolic basis
    # Use tanh for bounded, smooth transition (avoids overflow issues with exp)
    basis[:, 9] = np.tanh(x)
    
    # Compute prediction: dot product of basis with params
    # params is length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Safe computation: check for non-finite values
    pred = basis @ params
    
    # Replace any non-finite values with 0 to avoid breaking the optimizer
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
