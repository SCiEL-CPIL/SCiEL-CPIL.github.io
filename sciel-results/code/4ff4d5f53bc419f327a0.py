"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed basis geometry: 4 RBF centers and widths
_RBF_CENTERS = np.array([
    [-0.7, -0.7],
    [ 0.7, -0.7],
    [-0.7,  0.7],
    [ 0.7,  0.7],
])
_RBF_WIDTHS = np.array([0.45, 0.45, 0.45, 0.45])

# Precompute nothing here; compute on the fly for correctness with arbitrary x,y


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Polynomial part (6 terms)
    p = params
    pred = (
        p[0] * 1.0
        + p[1] * x
        + p[2] * y
        + p[3] * x * y
        + p[4] * x * x
        + p[5] * y * y
    )

    # RBF part (4 terms)
    # x, y are 1-D arrays of point coordinates
    for i in range(4):
        cx, cy = _RBF_CENTERS[i]
        w = _RBF_WIDTHS[i]
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        rbf = np.exp(-r2 / (w * w))
        pred = pred + p[6 + i] * rbf

    return pred
# EVOLVE-BLOCK-END
