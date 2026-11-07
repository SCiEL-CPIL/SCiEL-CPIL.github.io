"""Piecewise-linear hat basis for symbolic regression with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate y = f(x) on [-2, 2] using 10 piecewise-linear hat functions.
    
    Each hat is a triangular basis function centered at a knot, providing
    local flexibility without global oscillation. The representation is
    linear in params, so BFGS finds the global least-squares optimum.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # 10 uniform knots in [-2, 2]
    knots = np.linspace(-2.0, 2.0, 10)
    
    # Build hat basis matrix: each column is one hat function evaluated at x
    # Hat i: triangle peaking at knots[i], support [knots[i-1], knots[i+1]]
    # Boundary hats are one-sided
    basis = np.zeros((n, 10), dtype=np.float64)
    
    for i in range(10):
        if i == 0:
            # Left boundary: ramp up from knots[0] to knots[1]
            # Hat = (x - knots[0]) / (knots[1] - knots[0]) for x in [knots[0], knots[1]]
            # 0 elsewhere
            mask = (x >= knots[0]) & (x <= knots[1])
            basis[mask, i] = (x[mask] - knots[0]) / (knots[1] - knots[0])
        elif i == 9:
            # Right boundary: ramp down from knots[8] to knots[9]
            mask = (x >= knots[8]) & (x <= knots[9])
            basis[mask, i] = (knots[9] - x[mask]) / (knots[9] - knots[8])
        else:
            # Interior: triangle peaking at knots[i]
            # Left side: (x - knots[i-1]) / (knots[i] - knots[i-1]) for x in [knots[i-1], knots[i]]
            # Right side: (knots[i+1] - x) / (knots[i+1] - knots[i]) for x in [knots[i], knots[i+1]]
            mask_left = (x >= knots[i-1]) & (x <= knots[i])
            mask_right = (x >= knots[i]) & (x <= knots[i+1])
            basis[mask_left, i] = (x[mask_left] - knots[i-1]) / (knots[i] - knots[i-1])
            basis[mask_right, i] = (knots[i+1] - x[mask_right]) / (knots[i+1] - knots[i])
    
    result = basis @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
