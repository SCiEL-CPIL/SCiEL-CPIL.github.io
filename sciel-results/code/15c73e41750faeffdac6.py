"""Symbolic regression with 2 inputs: Grid-based RBF basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using a grid of Gaussian RBF centers.
    
    The 10 features are Gaussian kernels centered at 10 fixed points:
    - 4 corners: (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)
    - 4 edge midpoints: (0, -0.5), (0, 0.5), (-0.5, 0), (0.5, 0)
    - 1 center: (0, 0)
    - 1 additional: (0, 0) with different width
    
    This creates a localized basis that can capture spatial variations
    without the ill-conditioning issues of global polynomial bases.
    """
    n = x.shape[0]

    # Define 10 centers with different bandwidths
    centers = [
        (-0.5, -0.5, 8.0),
        (-0.5, 0.5, 8.0),
        (0.5, -0.5, 8.0),
        (0.5, 0.5, 8.0),
        (0.0, -0.5, 10.0),
        (0.0, 0.5, 10.0),
        (-0.5, 0.0, 10.0),
        (0.5, 0.0, 10.0),
        (0.0, 0.0, 6.0),
        (0.0, 0.0, 15.0),
    ]

    cols = []
    for cx, cy, sigma in centers:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        cols.append(np.exp(-sigma * r2))

    F = np.column_stack(cols)
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
