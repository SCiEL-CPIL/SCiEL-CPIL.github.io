"""
Strategy: Fixed-Knot Cubic Spline Model with Improved Knot Distribution.
The model is:
f(x) = sum_{i=0}^{9} c_i * B_i(x)

Where B_i(x) are cubic B-spline basis functions defined on the interval [-2, 2].
We use a proper B-spline knot vector with repeated boundary knots for clamping.

Parameters (length 10):
params[0:10] : Coefficients c_0 to c_9 for the spline basis.

This approach is linear in the parameters, making the optimization problem convex.
BFGS will reliably find the global minimum for the given basis.
"""
import numpy as np
from scipy.interpolate import BSpline

# Pre-compute the knot vector for the cubic spline.
# For a clamped cubic B-spline with 10 coefficients:
# We need n_knots = n_coeffs + degree = 10 + 3 = 13
# But for proper clamping, we repeat boundary knots (degree+1 times)
# Interior knots: 13 - 2*(3+1) = 5 interior knots
# Total: 4 (left) + 5 (interior) + 4 (right) = 13
# 
# Actually, let's use a simpler approach: uniform interior knots with 
# proper boundary handling via the BSpline constructor.
#
# For BSpline with degree k and n coefficients, we need n+k knots.
# To get a clamped spline, the first k+1 and last k+1 knots should be 
# at the boundaries.
#
# Let's use: 4 copies of -2, 5 interior points, 4 copies of 2
# Interior points: linspace(-2, 2, 7)[1:-1] gives 5 points
interior_knots = np.linspace(-2.0, 2.0, 7)[1:-1]  # 5 interior knots
KNOTS = np.concatenate([np.full(4, -2.0), interior_knots, np.full(4, 2.0)])
# KNOTS should have length 4 + 5 + 4 = 13
DEGREE = 3

# Pre-compute the basis matrix for faster evaluation
# We'll evaluate the basis at a set of points and use it as a linear model
# But since x varies, we need to compute on the fly.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    try:
        # Create the spline object
        spline = BSpline(KNOTS, params, DEGREE)
        
        # Evaluate at x, extrapolate=False to avoid issues outside domain
        result = spline(x, extrapolate=False)
        
        # Ensure finite output
        result = np.where(np.isfinite(result), result, 0.0)
        
        return result
    except Exception:
        # Fallback: use a simple polynomial basis as safety net
        # This ensures we always return something valid
        try:
            # Map x to [-1, 1] for better conditioning
            t = x / 2.0
            # Use monomial basis as fallback
            result = np.zeros_like(x)
            for i in range(min(10, len(params))):
                result += params[i] * (t ** i)
            return np.where(np.isfinite(result), result, 0.0)
        except:
            return np.zeros_like(x)
