# EVOLVE-BLOCK-START
import numpy as np
import scipy.interpolate

# Fixed knots for the spline representation
# Using 10 knots to utilize all 10 parameters as y-values at these knots
_KNOTS = np.linspace(-2.0, 2.0, 10)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Piecewise Cubic Spline.
    The 10 parameters represent the function values at 10 fixed, equally spaced
    knots in [-2, 2]. A natural cubic spline is constructed to interpolate
    these points.
    """
    # Ensure inputs are 1D arrays for consistency
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite params to avoid crashes in spline construction
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
        
    try:
        # Construct the natural cubic spline
        # bc_type='natural' sets the second derivative to zero at the boundaries
        spline = scipy.interpolate.CubicSpline(_KNOTS, params, bc_type='natural')
        
        # Evaluate the spline at x
        y_pred = spline(x)
        
        # Check for non-finite results
        if not np.all(np.isfinite(y_pred)):
             return np.zeros_like(x)
             
        return y_pred
    except Exception:
        # Fallback to a simple linear model if spline construction fails
        # This ensures we return a finite array even in edge cases
        return x * params[0] + params[1]
# EVOLVE-BLOCK-END
