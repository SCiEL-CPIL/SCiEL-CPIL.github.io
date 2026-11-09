"""Symbolic regression with 1 input using an RBF network structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a Radial Basis Function (RBF) network.
    
    The model is:
    y(x) = offset + sum_{k=1}^{5} weight_k * exp(-width * (x - center_k)^2)
    
    Parameters:
    params[0..4]: weights for the 5 RBF centers
    params[5]:    global offset (bias)
    params[6]:    width scaling factor (inverse of squared bandwidth)
    params[7..9]: unused (kept for interface compatibility, set to 0 effectively)
    
    The centers are fixed at equally spaced points in [-3, 3]:
    -3, -1.5, 0, 1.5, 3
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Clip extreme parameter values for numerical stability
    p = np.clip(p, -1e4, 1e4)
    
    # Fixed centers: 5 equally spaced points in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract parameters
    weights = p[0:5]
    offset = p[5]
    width = p[6]
    
    # Ensure width is positive to avoid issues with exp
    # If width is negative or zero, use a small positive value
    # We use abs(width) + small epsilon to ensure positive width
    width = abs(width) + 1e-6
    
    # Compute RBF activations
    # x shape: (N,), centers shape: (5,)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernels: exp(-width * diff^2)
    kernels = np.exp(-width * diff ** 2)
    
    # Linear combination of kernels + offset
    result = kernels @ weights + offset
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Safety margin: return if we're getting close to the time limit
    # (This is a simple guard; the computation is fast, but we check anyway)
    if time.time() - start_time > 10.0:
        return result
    
    return result
# EVOLVE-BLOCK-END
