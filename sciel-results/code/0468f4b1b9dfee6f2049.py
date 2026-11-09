"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import BSpline
import time

# Precompute the knot vector for a clamped cubic B-spline basis
# on the domain [-2, 2] with 10 basis functions.
# Degree k=3.
# Number of knots M = N + k + 1 = 10 + 3 + 1 = 14.
# We clamp the boundaries by repeating the first and last knots k+1 times.
# Interior knots are evenly spaced.
_N_PARAMS = 10
_DEGREE = 3
_KNOTS = np.concatenate([
    np.full(_DEGREE + 1, -2.0),       # 4 knots at -2
    np.linspace(-2.0, 2.0, 8)[1:-1],  # 6 interior knots
    np.full(_DEGREE + 1, 2.0)         # 4 knots at 2
])

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-Spline basis with fixed, evenly spaced interior knots.
    
    The basis is strictly linear in params, so BFGS converges to the
    global least-squares solution. Cubic B-Splines provide smooth ($C^2$),
    non-oscillatory approximation that is generally superior to linear
    hat functions for smooth targets due to higher order continuity and
    partition of unity properties.
    
    We use 10 cubic B-Splines on [-2, 2].
    """
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    # Create the BSpline object
    # t: knot vector
    # c: coefficients (params)
    # k: degree
    try:
        spline = BSpline(_KNOTS, params, _DEGREE)
        return spline(x)
    except Exception:
        # Fallback to zeros if something goes wrong (should not happen)
        return np.zeros_like(x)

# EVOLVE-BLOCK-END
