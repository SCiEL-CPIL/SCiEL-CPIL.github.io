"""Symbolic regression with 2 inputs: RBF network with 4 Gaussian centers."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function network with 4 Gaussian centers on a 2D grid.
    
    params = [a1, a2, a3, a4,  w1, w2, w3, w4,  cx, cy]
    
    Fixed centers at: (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)
    Widths are fixed at 1.0 for stability.
    Output: sum of a_i * exp(-||p - c_i||^2 / (2*w_i^2)) + cx*x + cy*y
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[8] * x + params[9] * y

    a1, a2, a3, a4, w1, w2, w3, w4, cx, cy = params

    # Fixed centers
    c1 = np.array([-0.5, -0.5])
    c2 = np.array([0.5, -0.5])
    c3 = np.array([-0.5, 0.5])
    c4 = np.array([0.5, 0.5])

    # Compute distances squared to each center
    dx1 = x - c1[0]
    dy1 = y - c1[1]
    dist1 = dx1**2 + dy1**2

    dx2 = x - c2[0]
    dy2 = y - c2[1]
    dist2 = dx2**2 + dy2**2

    dx3 = x - c3[0]
    dy3 = y - c3[1]
    dist3 = dx3**2 + dy3**2

    dx4 = x - c4[0]
    dy4 = y - c4[1]
    dist4 = dx4**2 + dy4**2

    # Gaussian kernels with fixed width 1.0
    g1 = np.exp(-dist1 / 2.0)
    g2 = np.exp(-dist2 / 2.0)
    g3 = np.exp(-dist3 / 2.0)
    g4 = np.exp(-dist4 / 2.0)

    # Weighted sum + linear terms
    pred = a1 * g1 + a2 * g2 + a3 * g3 + a4 * g4 + cx * x + cy * y

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
