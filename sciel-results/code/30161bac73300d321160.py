"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear hat basis for symbolic regression with 10 parameters.
    
    The key insight from the inspiration: the problem is linear in parameters.
    A piecewise-linear basis (hat functions) can represent arbitrary piecewise-
    linear functions exactly. With 10 parameters, we can create a flexible
    approximation that adapts to the local shape of the target function.
    
    We use a hat basis on a fixed grid over [-2, 2]. The 10 parameters are
    the coefficients of 10 hat functions centered at evenly spaced points.
    
    This is linear in params, so BFGS will converge quickly (it's effectively
    a linear least-squares problem). The hat basis is local, so it can capture
    sharp features, discontinuities, and non-smooth behavior that global
    polynomial bases cannot.
    """
    # Fixed grid for hat centers: 10 points evenly spaced in [-2, 2]
    # Centers at -1.7, -1.3, -0.9, -0.5, -0.1, 0.3, 0.7, 1.1, 1.5, 1.9
    # This gives good coverage of the domain [-2, 2]
    
    # Define hat function centers
    centers = np.array([-1.7, -1.3, -0.9, -0.5, -0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    
    # Half-width of each hat function
    # Adjacent centers are 0.4 apart, so half-width = 0.2
    # This makes adjacent hats just touch at their edges
    half_width = 0.2
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        c = centers[i]
        # Hat function: triangular bump centered at c with half-width hw
        # f(x) = max(0, 1 - |x - c| / hw)
        diff = np.abs(x - c)
        hat = np.maximum(0.0, 1.0 - diff / half_width)
        result += params[i] * hat
    
    return result
# EVOLVE-BLOCK-END
