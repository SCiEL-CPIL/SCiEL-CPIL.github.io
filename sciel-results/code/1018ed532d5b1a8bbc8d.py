"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute fixed centers for the RBF basis
# 8 centers distributed in [-1, 1]^2
CENTERS = np.array([
    [-0.7, -0.7],
    [-0.7,  0.7],
    [ 0.7, -0.7],
    [ 0.7,  0.7],
    [ 0.0,  0.0],
    [-0.7,  0.0],
    [ 0.7,  0.0],
    [ 0.0,  0.7]
])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # If params are invalid or NaN, return zeros or linear fallback to avoid crash
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
        
    # Extract parameters
    weights = params[0:8]
    log_sigma = params[8]
    bias = params[9]
    
    # Compute sigma. Clamp log_sigma to prevent overflow/underflow in exp
    # log_sigma in [-5, 5] -> sigma in [0.006, 148]
    log_sigma_clamped = np.clip(log_sigma, -5.0, 5.0)
    sigma = np.exp(log_sigma_clamped)
    
    # Compute distances from each point (x, y) to each center
    # points shape: (N, 2)
    points = np.column_stack((x, y))
    
    # distances shape: (N, 8)
    # (points[:, np.newaxis, :] - CENTERS[np.newaxis, :, :])**2
    diffs = points[:, np.newaxis, :] - CENTERS[np.newaxis, :, :]
    sq_dist = np.sum(diffs ** 2, axis=2)
    
    # RBF values: exp(-sq_dist / (2 * sigma^2))
    # Avoid division by zero if sigma is tiny (though clamped)
    rbf_values = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Weighted sum
    pred = np.sum(rbf_values * weights[np.newaxis, :], axis=1) + bias
    
    # Safety check for non-finite outputs
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    # Time guard (unlikely to trigger for 200 points, but required by directive)
    if time.time() - start_time > 100:
         pass # We already computed it, so just return
        
    return pred
# EVOLVE-BLOCK-END
