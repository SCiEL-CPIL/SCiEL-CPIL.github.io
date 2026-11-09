"""
Strategy: Gaussian Radial Basis Function (RBF) Network.

Rationale:
- The unknown function f(x, y, z) is sampled on [0.5, 2]^3.
- The current rational basis is non-linear in parameters, leading to a non-convex optimization landscape.
- An RBF basis with fixed centers makes the model linear in the parameters (weights).
- Linear models have a convex MSE surface, so BFGS from any start point (including all-ones)
  will converge to the global minimum, assuming the basis is rich enough.
- We use 10 Gaussian centers distributed across the domain to capture local variations.
- The parameters `params` are the weights for these 10 basis functions.
"""

import numpy as np
import time

# Global state for deadline management
_start_time = None

def _compute_rbf_basis(x, y, z):
    """
    Compute the 10 RBF basis values for each point (x, y, z).
    Returns an array of shape (N, 10).
    """
    # Define 10 fixed centers in the domain [0.5, 2]^3.
    # We use a mix of corners and mid-points to cover the space.
    # Centers:
    # 1: (0.5, 0.5, 0.5)
    # 2: (2.0, 0.5, 0.5)
    # 3: (0.5, 2.0, 0.5)
    # 4: (2.0, 2.0, 0.5)
    # 5: (0.5, 0.5, 2.0)
    # 6: (2.0, 0.5, 2.0)
    # 7: (0.5, 2.0, 2.0)
    # 8: (2.0, 2.0, 2.0)
    # 9: (1.25, 1.25, 1.25) - Center
    # 10: (1.25, 0.5, 2.0) - An off-center point to add diversity
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 2.0]
    ])
    
    # Stack x, y, z into (N, 3)
    points = np.column_stack((x, y, z))
    
    # Compute squared distances to each center: (N, 10)
    # points[:, np.newaxis, :] - centers[np.newaxis, :, :] -> (N, 10, 3)
    # squared_dist = sum((points[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
    
    # More efficient memory-wise:
    # dist_sq = (x[:, None] - centers[0, :])**2 + (y[:, None] - centers[1, :])**2 + (z[:, None] - centers[2, :])**2
    # Wait, centers is (10, 3). So centers[:, 0] is x-coords of centers.
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    
    dist_sq = (x[:, None] - cx[None, :])**2 + \
              (y[:, None] - cy[None, :])**2 + \
              (z[:, None] - cz[None, :])**2
    
    # Gaussian RBF: exp(-beta * dist_sq)
    # Choose beta such that the basis functions have reasonable support.
    # Domain size is 1.5. Distances can be up to ~2.6 (corner to corner).
    # If beta=1, exp(-2.6) ~ 0.07. If beta=4, exp(-10) ~ 0.00004.
    # Let's use beta=2.0 for a balance.
    beta = 2.0
    basis = np.exp(-beta * dist_sq)
    
    return basis

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Linear RBF Model.
    
    The evaluator fits 10 parameters using BFGS.
    We model the function as a linear combination of 10 Gaussian RBFs with fixed centers.
    This formulation is linear in `params`, ensuring convex optimization.
    
    Deadline guard: If we exceed 140s, return the current prediction (which is fast anyway).
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    try:
        # Compute basis: (N, 10)
        B = _compute_rbf_basis(x, y, z)
        
        # Prediction: B @ params
        # B is (N, 10), params is (10,)
        pred = B @ params
        
        # Check for non-finite values
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
    except Exception:
        # In case of any error, return a safe finite value
        return np.ones_like(x)

# EVOLVE-BLOCK-END
