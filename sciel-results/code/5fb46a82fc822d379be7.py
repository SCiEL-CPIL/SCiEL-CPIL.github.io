"""Symbolic regression with 2 inputs - Radial Basis Function (Gaussian) basis, optimized centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Radial Basis Function (RBF) basis with Gaussian kernels.
    
    We use 9 Gaussian centers on a 3x3 grid centered at [-0.5, 0, 0.5] and 1 constant bias term.
    Total 10 parameters.
    
    f(x, y) = p0 + sum(p_i * exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma^2)))
    
    The centers are shifted inward from the boundary to improve coverage of the entire domain 
    and reduce boundary artifacts. The bandwidth is tuned for good overlap.
    """
    # Check time
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a safe fallback (e.g., mean of y if available, or 0)
        # Since we don't have y in the prediction phase for scoring (only x), 
        # but during fitting y is present. 
        # If we time out during fitting, it's an error. 
        # This guard is mainly for the evaluation loop if it calls equation repeatedly? 
        # The task says "Each evaluation must finish within 120s". 
        # BFGS calls equation many times. 
        # We cannot easily return "best result" from inside equation without state.
        # So we just proceed, but keep computation fast.
        pass

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64) # y is not used in prediction, but present in signature
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure at least 10 params
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
        
    p0 = p[0]
    p_coeffs = p[1:10]
    
    # Centers for the 9 Gaussian kernels
    # Grid: -0.5, 0, 0.5 (shifted inward from -1, 0, 1)
    cx = np.array([-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    cy = np.array([-0.5,  0.0,  0.5, -0.5, 0.0, 0.5, -0.5, 0.0,  0.5])
    
    # Bandwidth
    # Increased slightly to 0.7 to ensure good overlap with the shifted centers
    sigma = 0.7
    sigma2 = 2.0 * sigma * sigma
    
    # Calculate squared distances
    # x shape: (N, 1) or (N,)
    # Ensure x is column vector for broadcasting
    if x.ndim == 1:
        x_col = x[:, np.newaxis]
    else:
        x_col = x
    
    if y.ndim == 1:
        y_col = y[:, np.newaxis]
    else:
        y_col = y
        
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    dist2 = dx**2 + dy**2
    
    # Compute Gaussian weights
    # Shape: (N, 9)
    weights = np.exp(-dist2 / sigma2)
    
    # Linear combination
    result = p0 + weights @ p_coeffs
    
    return result
# EVOLVE-BLOCK-END
