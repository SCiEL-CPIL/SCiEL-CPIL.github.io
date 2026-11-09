"""Symbolic regression with 1 input(s) using a hybrid basis with piecewise linear hat functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - Low-order polynomials (1, x, x^2) for global trends.
    - Piecewise linear hat functions at 6 evenly spaced knots for local flexibility.
    
    The hat functions create a continuous but non-smooth function class that can
    approximate sharp transitions while remaining linear-in-parameters for stable BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the hybrid basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # Column 0: constant term (bias)
    basis[:, 0] = 1.0
    
    # Columns 1-2: polynomial terms x, x^2
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    
    # Columns 3-8: Piecewise linear hat functions at 6 evenly spaced knots
    # Domain is [0.1, 6], so knots are at 0.1, 1.25, 2.4, 3.55, 4.7, 5.85
    # Actually, let's use 6 internal knots evenly spaced in [0.1, 6]
    # We'll create hat functions centered at these knots with support width = spacing
    domain_min = 0.1
    domain_max = 6.0
    n_knots = 6
    knot_spacing = (domain_max - domain_min) / (n_knots + 1)
    
    for i in range(n_knots):
        center = domain_min + (i + 1) * knot_spacing
        half_width = knot_spacing
        
        # Hat function: linear rise from center-half_width to center, 
        # linear fall from center to center+half_width
        # h(x) = max(0, 1 - |x - center| / half_width)
        dist = np.abs(x - center)
        hat = np.maximum(0.0, 1.0 - dist / half_width)
        basis[:, 3 + i] = hat
    
    # Column 9: exponential decay term for potential asymptotic behavior
    basis[:, 9] = np.exp(-x)
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
