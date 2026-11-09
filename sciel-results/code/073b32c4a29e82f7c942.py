"""Symbolic regression with 2 inputs - Radial Basis Function (Gaussian) basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Radial Basis Function (RBF) basis with Gaussian kernels.
    
    We use 9 Gaussian centers on a 3x3 grid over [-1, 1] and 1 constant bias term.
    Total 10 parameters.
    
    f(x, y) = p0 + sum(p_i * exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma^2)))
    
    This basis is local, allowing it to fit sharp features and non-polynomial shapes
    that a global cubic polynomial might miss or fit poorly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure at least 10 params
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
        
    p0 = p[0]
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Centers for the 9 Gaussian kernels
    # Grid: -1, 0, 1
    cx = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    cy = np.array([-1.0,  0.0,  1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    
    # Bandwidth
    sigma = 0.6
    sigma2 = 2.0 * sigma * sigma
    
    # Calculate squared distances
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    dist2 = dx**2 + dy**2
    
    # Compute Gaussian weights
    # Shape: (N, 9)
    weights = np.exp(-dist2 / sigma2)
    
    # Linear combination
    # p[1:10] corresponds to the 9 centers
    result = p0 + weights @ p[1:10]
    
    return result
# EVOLVE-BLOCK-END
