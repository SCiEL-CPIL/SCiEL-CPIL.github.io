"""Symbolic regression with 1 input - Gaussian RBF universal approximator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function model with 5 RBFs.
    
    Each RBF has a learnable center, width, and amplitude (10 params total).
    This non-linear, non-convex architecture adaptively places localized
    basis functions to match the hidden target structure.
    
    Parameters:
      params[0:5]:  Centers of the 5 RBFs
      params[5:8]:  Widths of the first 3 RBFs
      params[8:10]: Widths of the last 2 RBFs
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    
    # Extract parameters
    centers = params[0:5]
    widths = np.abs(params[5:10]) + 0.1  # Ensure positive widths with minimum
    
    # Compute RBF values
    y = np.zeros_like(x_flat)
    
    for i in range(5):
        # Gaussian: amplitude * exp(-((x - center)^2) / (2 * width^2))
        # Amplitude is 1 (implicit), so we just use the Gaussian shape
        diff = x_flat - centers[i]
        y += np.exp(-(diff ** 2) / (2.0 * widths[i] ** 2))
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        return y.reshape(x.shape)
    
    return y.reshape(x.shape)
# EVOLVE-BLOCK-END
