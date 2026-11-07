"""
Strategy: Gaussian RBF Network with Fixed Centers.
Instead of a linear combination of polynomial basis functions, we use a linear combination of Gaussian RBFs.
The centers are fixed to cover the domain [0.5, 2]^3.
The widths are fixed to provide local coverage.
The parameters are the weights for each RBF.
This allows the model to fit non-polynomial structures (like exponentials or trigonometric functions) 
better than a fixed quadratic polynomial, provided the hidden function is smooth.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Define fixed centers for the RBFs.
    # Domain is [0.5, 2.0]. Let's use 10 centers.
    # To cover the volume, we can distribute them.
    # A simple grid might be too sparse. A Latin Hypercube or specific set is better.
    # Let's use a set of points that covers the corners and center.
    
    # Centers: (x_c, y_c, z_c)
    # 1. Center of domain: (1.25, 1.25, 1.25)
    # 2-5. Mid-points of faces
    # 6-9. Corners
    # 10. Another interior point?
    
    # Actually, let's use a more systematic approach.
    # We have 10 params.
    # Let's use 10 Gaussian RBFs.
    # Width sigma: If domain size is 1.5, a sigma of ~0.5 gives good local coverage.
    
    # Fixed Centers (hardcoded for stability and speed)
    # We want to span the space [0.5, 2.0] in all 3 dimensions.
    # Let's pick 10 points roughly uniformly distributed.
    
    # Using a simple heuristic:
    # 1: (1.0, 1.0, 1.0)
    # 2: (1.5, 1.0, 1.0)
    # 3: (1.0, 1.5, 1.0)
    # 4: (1.0, 1.0, 1.5)
    # 5: (1.5, 1.5, 1.0)
    # 6: (1.5, 1.0, 1.5)
    # 7: (1.0, 1.5, 1.5)
    # 8: (1.5, 1.5, 1.5)
    # 9: (0.7, 0.7, 0.7)
    # 10: (1.8, 1.8, 1.8)
    
    centers = np.array([
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [1.0, 1.5, 1.0],
        [1.0, 1.0, 1.5],
        [1.5, 1.5, 1.0],
        [1.5, 1.0, 1.5],
        [1.0, 1.5, 1.5],
        [1.5, 1.5, 1.5],
        [0.7, 0.7, 0.7],
        [1.8, 1.8, 1.8]
    ])
    
    # Widths: Fixed sigma for all RBFs.
    # If sigma is too small, the problem becomes ill-conditioned (spiky).
    # If sigma is too large, it becomes like a low-order polynomial.
    # Domain range is 1.5. A sigma of 0.8 seems reasonable for smooth interpolation.
    sigma = 0.8
    sigma2 = 2 * sigma * sigma  # 2 * sigma^2 for the exponent
    
    # Stack inputs into a matrix of shape (N, 3)
    inputs = np.column_stack((x, y, z))
    
    # Calculate distances to centers
    # inputs: (N, 3)
    # centers: (10, 3)
    # We need distance between each input point and each center.
    # dist^2 = sum((inputs - centers)^2)
    
    # Use broadcasting: inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # Shape: (N, 10, 3)
    diffs = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2) # Shape: (N, 10)
    
    # Calculate RBF values: exp(-dists_sq / (2 * sigma^2))
    # To avoid underflow/overflow issues, we rely on the fact that dists_sq is positive.
    # For large distances, exp(-large) -> 0, which is fine.
    rbf_vals = np.exp(-dists_sq / sigma2)
    
    # rbf_vals shape: (N, 10)
    # params shape: (10,)
    
    # Prediction: sum over RBFs
    # pred = rbf_vals @ params
    pred = np.sum(rbf_vals * params, axis=1)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x, dtype=np.float64)
        
    return pred
# EVOLVE-BLOCK-END
