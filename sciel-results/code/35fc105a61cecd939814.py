"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global piecewise linear "hat function" basis with fixed knots.
    
    The basis is strictly linear in params, so BFGS converges to the
    global least-squares solution. Hat functions provide flexible,
    non-oscillatory approximation that can capture arbitrary continuous
    shapes without local minima traps of nonlinear models.
    
    We use 10 hat functions with evenly spaced knots across [-2, 2].
    Each hat is a triangular bump centered at a knot location.
    params[0:10] -> weights for the 10 hat functions
    """
    # Define 10 evenly spaced knot centers across [-2, 2]
    n_hats = 10
    centers = np.linspace(-1.8, 1.8, n_hats)
    
    # Half-width of each hat function (distance from center to zero crossing)
    # With 10 centers over range 3.6, spacing is 0.4, so half-width = 0.2
    # But we want overlapping hats for smooth interpolation, so use half-width = 0.4
    half_width = 0.4
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(n_hats):
        c = centers[i]
        # Hat function: max(0, 1 - |x - c| / half_width)
        # This is 1 at the center, 0 at distance half_width, and 0 beyond
        dist = np.abs(x - c)
        hat = np.maximum(0.0, 1.0 - dist / half_width)
        result += params[i] * hat
    
    return result
# EVOLVE-BLOCK-END
