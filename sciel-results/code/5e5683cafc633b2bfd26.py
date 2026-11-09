"""Symbolic regression with 1 input(s) using piecewise linear hat functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of 10 piecewise
    linear hat functions (B-spline basis of order 1).
    
    The domain [0.1, 6] is divided into 9 equal intervals with 10 interior
    knots. Each hat function is continuous, piecewise linear, and non-negative,
    peaking at 1.0 at its corresponding knot and zero elsewhere. This basis
    can approximate discontinuities and sharp transitions that smooth global
    bases (polynomials, RBFs) struggle with, while remaining linear-in-params
    for stable BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Define 10 evenly spaced knots across [0.1, 6]
    x_min, x_max = 0.1, 6.0
    n_knots = 10
    knots = np.linspace(x_min, x_max, n_knots)
    
    # Build the hat function basis matrix
    # Shape: (n_points, n_knots)
    basis = np.zeros((n_points, n_knots), dtype=np.float64)
    
    # For each knot i, compute the hat function value at each x
    # Hat function centered at knots[i]:
    #   - Left neighbor: knots[i-1] (or x_min if i=0)
    #   - Right neighbor: knots[i+1] (or x_max if i=n_knots-1)
    #   - Value = max(0, 1 - |x - knots[i]| / half_width)
    #     where half_width = min(knots[i] - left, right - knots[i])
    
    for i in range(n_knots):
        if i == 0:
            left = x_min
            right = knots[1]
        elif i == n_knots - 1:
            left = knots[n_knots - 2]
            right = x_max
        else:
            left = knots[i - 1]
            right = knots[i + 1]
        
        center = knots[i]
        half_width = min(center - left, right - center)
        
        if half_width <= 0:
            basis[:, i] = 1.0  # degenerate case
        else:
            # Triangular hat: 1 at center, linearly decreasing to 0 at edges
            basis[:, i] = np.clip(1.0 - np.abs(x - center) / half_width, 0.0, 1.0)
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
