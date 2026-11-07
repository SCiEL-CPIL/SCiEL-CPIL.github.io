"""Rewritten program for symbolic regression with RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF basis model with 10 parameters.
    f(x, y) = p0 + sum_{i=1}^{9} p_i * exp(-gamma * ((x - c_xi)^2 + (y - c_yi)^2))
    
    Centers are at a 3x3 grid: {-0.5, 0, 0.5} x {-0.5, 0, 0.5}
    Gamma is fixed at 4.0 for good resolution on [-1,1]^2.
    
    This is a universal approximator for smooth functions and has a convex-ish
    loss landscape for BFGS (linear in params), avoiding local minima traps
    of rational or high-degree polynomial bases.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    p = params

    # Fixed gamma for RBF width
    gamma = 4.0

    # 9 RBF centers on a 3x3 grid
    # Order: (-0.5,-0.5), (-0.5,0), (-0.5,0.5), (0,-0.5), (0,0), (0,0.5), (0.5,-0.5), (0.5,0), (0.5,0.5)
    centers = [
        (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5),
        (0.0, -0.5), (0.0, 0.0), (0.0, 0.5),
        (0.5, -0.5), (0.5, 0.0), (0.5, 0.5)
    ]

    # Compute RBF values
    # pred = p0 + p1*RBF1 + ... + p9*RBF9
    pred = np.full_like(x, p[0], dtype=np.float64)

    for i in range(9):
        cx, cy = centers[i]
        dist_sq = (x - cx)**2 + (y - cy)**2
        rbf_val = np.exp(-gamma * dist_sq)
        pred += p[i+1] * rbf_val

    if time.time() > deadline:
        return pred

    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
# EVOLVE-BLOCK-END
