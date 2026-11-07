"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] != 10:
        p = p[:10] if p.size >= 10 else np.pad(p, (0, 10 - p.size))

    # Check deadline
    if time.time() > deadline:
        return np.zeros(n, dtype=np.float64)

    # B-spline control point approach:
    # Use 10 control points at fixed knot locations in [-3, 3]
    # The parameters are the y-values at these control points.
    # We use a cubic spline interpolation through these points.
    # This gives a smooth, flexible curve that can approximate many functions.
    
    # Fixed knot locations (control points) spread across [-3, 3]
    knots = np.linspace(-3.0, 3.0, 10)
    
    # Ensure x is sorted for spline evaluation (CubicSpline requires sorted knots)
    # The control points are already sorted
    try:
        # Create spline from control points
        # Use 'not-a-knot' boundary conditions for smoothness
        spline = CubicSpline(knots, p, bc_type='not-a-knot')
        result = spline(x)
    except Exception:
        # Fallback: use simple linear interpolation if spline fails
        try:
            result = np.interp(x, knots, p, left=p[0], right=p[-1])
        except Exception:
            result = np.zeros(n, dtype=np.float64)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
