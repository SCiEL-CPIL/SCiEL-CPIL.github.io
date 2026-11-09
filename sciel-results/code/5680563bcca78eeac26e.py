"""Symbolic regression with 2 inputs: Radial Basis Function (RBF) network with fixed centers."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0

# Fixed centers for RBFs
# 10 centers distributed across the domain [-1, 1] x [-1, 1]
# Using a 2x5 grid to cover the domain more uniformly than boundary-only points
_x_coords = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
_y_coords = np.array([-0.5, 0.5])
_CENTERS = np.array([(x, y) for y in _y_coords for x in _x_coords])

_SIGMA2 = 2 * (0.5 ** 2)  # 2 * sigma^2 for the exponent


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute RBF features for each point.
    Returns a (n, 10) matrix where F[i, j] = exp(-||p_i - c_j||^2 / (2*sigma^2)).
    """
    # Stack x and y into points (n, 2)
    points = np.column_stack((x, y))
    
    # Compute squared distances to each center
    # points: (n, 1, 2)
    # centers: (1, 10, 2)
    # diff: (n, 10, 2)
    diff = points[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]
    
    # Squared Euclidean distance: sum over last axis
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Gaussian kernel
    F = np.exp(-dist_sq / _SIGMA2)
    
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute predictions as linear combination of 10 RBF features.
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
