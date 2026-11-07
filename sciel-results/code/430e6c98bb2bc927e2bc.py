"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using cubic B-spline basis with 10 control points.
    
    The 10 parameters are the B-spline control values at knot positions
    evenly spaced across [-3, 3]. The basis functions are cubic B-splines
    with uniform knots, providing smooth piecewise-polynomial interpolation.
    
    This gives high local flexibility: each parameter directly influences
    the curve shape in its neighborhood, allowing the fitter to adapt to
    arbitrary smooth functions.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)

    # Define 10 knot positions evenly spaced in [-3, 3]
    # For cubic B-splines, we need 4 extra knots for the boundary
    # Knot vector: [t0, t0, t0, t0, k1, k2, ..., k9, t10, t10, t10, t10]
    # where t0 = -3, t10 = 3, and k1..k9 are internal knots
    
    # Internal knots: 9 points evenly spaced in [-3, 3]
    # Actually for 10 control points and cubic splines, we need:
    # Number of knots = n_ctrl + degree + 1 = 10 + 3 + 1 = 14
    # Uniform knots: from -3 to 3 with 13 intervals
    # Knots: -3, -3, -3, -3, -1.8, -0.6, 0.6, 1.8, 3, 3, 3, 3
    # Wait, let me use a standard uniform B-spline setup.
    
    # For 10 control points with cubic B-splines:
    # We use knots at: -3, -3, -3, -3, -1.8, -0.6, 0.6, 1.8, 3, 3, 3, 3
    # That's 12 knots, but we need 14. Let me recalculate.
    
    # Standard: n_ctrl = 10, degree = 3
    # Total knots = n_ctrl + degree + 1 = 14
    # Uniform knot spacing: (3 - (-3)) / (n_ctrl - 1) = 6/9 = 2/3
    # Knots: -3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 3
    # That's 13 knots. Need one more.
    
    # Let me use: knots = [-3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 3, 3]
    # That's 14 knots.
    
    knots = np.array([-3.0, -3.0, -3.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    
    # Compute cubic B-spline basis functions N_{i,3}(x) for i = 0..9
    # Using de Boor's algorithm / Cox-de Boor recursion
    
    def bspline_basis(x_vals, knots, i, degree):
        """Compute B-spline basis function N_{i,degree}(x) for array x_vals."""
        if degree == 0:
            return np.where((x_vals >= knots[i]) & (x_vals < knots[i+1]), 1.0, 0.0)
        
        left = np.zeros_like(x_vals, dtype=np.float64)
        right = np.zeros_like(x_vals, dtype=np.float64)
        
        # Left term
        denom_left = knots[i + degree] - knots[i]
        if denom_left > 1e-15:
            left = (x_vals - knots[i]) / denom_left * bspline_basis(x_vals, knots, i, degree - 1)
        
        # Right term
        denom_right = knots[i + degree + 1] - knots[i + 1]
        if denom_right > 1e-15:
            right = (knots[i + degree + 1] - x_vals) / denom_right * bspline_basis(x_vals, knots, i + 1, degree - 1)
        
        return left + right
    
    # Compute all 10 basis functions
    basis = np.zeros((n, 10), dtype=np.float64)
    for i in range(10):
        basis[:, i] = bspline_basis(x, knots, i, 3)
    
    # For points at x = 3 (right boundary), ensure the last basis function is 1
    # due to the half-open interval convention
    right_mask = (x >= 3.0 - 1e-12)
    if np.any(right_mask):
        basis[right_mask, :] = 0.0
        basis[right_mask, 9] = 1.0
    
    # Compute result as linear combination
    result = basis @ p
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
