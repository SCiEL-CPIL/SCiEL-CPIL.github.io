"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network for nonlinear oscillator acceleration.
    
    Uses 10 RBFs with fixed centers and widths, and 10 fitted weights.
    Inputs are normalized to [-1, 1] range for consistent RBF scaling.
    """
    t0 = time.time()
    
    # Safety guard
    if time.time() - t0 > 140.0:
        return np.zeros_like(t, dtype=np.float64)
    
    # Normalize inputs to [-1, 1] approx
    tn = (t - 40.0) / 10.0
    xn = x / 0.35
    vn = v / 0.19
    
    # Define 10 centers in normalized space
    # Covering the cube [-1, 1]^3
    centers = np.array([
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0,  1.0]
    ])
    
    # Extract center coordinates for vectorized computation
    ct = centers[:, 0]
    cx = centers[:, 1]
    cv = centers[:, 2]
    
    # Reshape inputs for broadcasting (N, 1)
    tn_e = tn[:, np.newaxis]
    xn_e = xn[:, np.newaxis]
    vn_e = vn[:, np.newaxis]
    
    # Compute squared distances (N, 10)
    dt = tn_e - ct
    dx = xn_e - cx
    dv = vn_e - cv
    dist_sq = dt**2 + dx**2 + dv**2
    
    # Gaussian RBF activation
    sigma = 0.8
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Linear combination of RBFs
    result = rbf_vals @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
        
    return result
# EVOLVE-BLOCK-END
