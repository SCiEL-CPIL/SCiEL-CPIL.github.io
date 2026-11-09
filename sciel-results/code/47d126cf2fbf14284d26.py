"""RBF basis with fixed centers for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute RBF centers and bandwidth
# 10 centers: 8 corners of [0.5, 2]^3 + 2 mid-z points
_cx = np.array([0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25])
_cy = np.array([0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 1.25, 1.25])
_cz = np.array([0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0])
# Centers shape: (10, 3)
_centers = np.stack([_cx, _cy, _cz], axis=1)

# Bandwidth: chosen so RBFs have reasonable local support
# Domain range is 1.5 per axis, so a bandwidth of ~0.5 gives good locality
_GAMMA = 2.0  # exp(-gamma * dist^2), with dist in original units


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based prediction with 10 fixed centers.
    params: length-10 array of coefficients for each RBF.
    """
    # Stack inputs into (n, 3)
    pts = np.stack([x, y, z], axis=1)  # (n, 3)
    
    # Compute squared distances from each point to each center
    # diff: (n, 10, 3)
    diff = pts[:, np.newaxis, :] - _centers[np.newaxis, :, :]  # (n, 10, 3)
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # RBF values: exp(-gamma * sq_dist)
    rbf = np.exp(-_GAMMA * sq_dist)  # (n, 10)
    
    # Linear combination with params
    return rbf @ params
# EVOLVE-BLOCK-END
