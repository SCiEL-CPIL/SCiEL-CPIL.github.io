"""Symbolic regression with 2 inputs: Localized RBF network with Gaussian kernels."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before the 150s kill

# Precompute RBF centers on a 3x3 grid over [-1,1]^2
_CENTERS = np.array([
    [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
    [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
    [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
    [0.0,  0.0],  # duplicate center for flexibility
])
_K_WIDTH = 0.4  # Gaussian kernel width parameter


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix: Gaussian RBFs at 10 fixed centers.
    
    Each feature is exp(-||[x,y] - c_i||^2 / (2 * sigma^2)) for center c_i.
    Localized basis functions avoid global polynomial ill-conditioning.
    """
    # Stack input coordinates: (n, 2)
    pts = np.column_stack([x, y])  # (n, 2)
    
    # Compute squared distances to each center: (n, 10)
    # centers: (10, 2), pts: (n, 2)
    dx = pts[:, 0:1] - _CENTERS[:, 0:1].T  # (n, 10)
    dy = pts[:, 1:2] - _CENTERS[:, 1:2].T  # (n, 10)
    dist_sq = dx * dx + dy * dy  # (n, 10)
    
    # Gaussian kernel: exp(-dist_sq / (2 * sigma^2))
    sigma_sq = _K_WIDTH * _K_WIDTH
    F = np.exp(-dist_sq / (2.0 * sigma_sq))
    
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute predictions as weighted sum of 10 Gaussian RBF features.
    
    The evaluator fits params via BFGS to minimize training MSE.
    This function must be fast and numerically stable.
    """
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred

    F = _features(x, y)

    # Compute prediction
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
