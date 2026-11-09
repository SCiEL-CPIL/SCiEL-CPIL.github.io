"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a flexible, normalized basis of 10 functions.
    The evaluator fits params (length 10) via BFGS starting from all ones.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    n = x.shape[0]
    
    # Build a rich basis of 10 candidate features
    # We normalize each basis function to have unit RMS to improve conditioning
    basis_list = [
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        np.sin(x),
        np.cos(x),
        np.tanh(x),
        np.exp(-0.5 * x ** 2),  # Gaussian bump
        np.exp(0.5 * x),        # Exponential growth (bounded domain)
        x * np.sin(x),
    ]
    
    # Stack into a matrix (n, 10)
    B = np.column_stack(basis_list)
    
    # Normalize each column to unit RMS to improve Hessian conditioning
    # RMS = sqrt(mean(b_i^2))
    rms = np.sqrt(np.mean(B ** 2, axis=0))
    # Avoid division by zero
    rms[rms == 0] = 1.0
    B = B / rms
    
    # Prediction: B @ params
    # params is length 10
    if params.shape[0] != 10:
        # Safety: if params length is wrong, return zeros (should not happen)
        return np.zeros_like(x)
        
    pred = B @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
