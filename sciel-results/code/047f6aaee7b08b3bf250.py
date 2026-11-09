"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 150s limit, keep 5s margin

# Pre-compute centers and bandwidth
# Inputs x, y, z are in [0.5, 2.0]. Normalize to [0, 1].
# Centers in [0, 1]^3
# 8 corners + 2 face centers
_CENTERS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5]
])
_GAMMA = 4.0  # Bandwidth parameter for exp(-gamma * dist^2)


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF basis with 10 fixed centers in the normalized domain [0, 1]^3.
    Centers: 8 corners of the cube + 2 face centers.
    Kernel: exp(-gamma * ||x - c_i||^2)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe, finite array. 
        # The optimizer might still try to step, but we prevent long hangs.
        # Returning zeros is a safe fallback, though likely to score poorly.
        return np.zeros_like(x, dtype=float)

    # Normalize inputs from [0.5, 2.0] to [0, 1]
    u = (x - 0.5) / 1.5
    v = (y - 0.5) / 1.5
    w = (z - 0.5) / 1.5
    
    # Compute squared distances to each center
    # dist^2 = (u - cu)^2 + (v - cv)^2 + (w - cw)^2
    # We can do this efficiently using broadcasting
    # u, v, w are 1D arrays of shape (N,)
    # _CENTERS is (10, 3)
    
    # Shape (N, 1)
    uu = u[:, np.newaxis]
    vv = v[:, np.newaxis]
    ww = w[:, np.newaxis]
    
    # Shape (1, 10)
    cu = _CENTERS[:, 0][np.newaxis, :]
    cv = _CENTERS[:, 1][np.newaxis, :]
    cw = _CENTERS[:, 2][np.newaxis, :]
    
    # Squared distances shape (N, 10)
    dist_sq = (uu - cu)**2 + (vv - cv)**2 + (ww - cw)**2
    
    # RBF values shape (N, 10)
    rbf_vals = np.exp(-_GAMMA * dist_sq)
    
    # Linear combination
    # params shape (10,)
    # rbf_vals @ params -> shape (N,)
    return rbf_vals @ params
# EVOLVE-BLOCK-END
