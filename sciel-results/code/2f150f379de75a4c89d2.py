# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses numpy's polynomial fitting which handles scaling internally
    via SVD, providing a stable least-squares solution.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle potential NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    try:
        # polyfit returns coefficients in ascending order: c_0, c_1, ..., c_5
        # This matches the required output format: sum(c_i * x**i)
        c = P.polyfit(x, y, 5)
        c = np.asarray(c, dtype=np.float64)
        
        # Check for finiteness
        if not np.all(np.isfinite(c)):
            return np.zeros(6)
            
        return c
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
