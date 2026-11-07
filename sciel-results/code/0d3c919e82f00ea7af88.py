# EVOLVE-BLOCK-START
import numpy as np
import time


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

    # Cubic B-spline basis with 10 control points on [-3, 3]
    # This gives a smooth, flexible curve that can approximate many functions.
    # The 10 parameters are the y-values at fixed knot locations.
    
    # Fixed knot locations (control points) spread across [-3, 3]
    knots = np.linspace(-3.0, 3.0, 10)
    
    # Build cubic B-spline basis matrix
    # For each x, compute the contribution from each control point
    # Using the standard cubic B-spline basis functions
    
    # We use a simple approach: for each control point, compute a cubic
    # basis function that is non-zero in a local neighborhood
    # This is equivalent to using piecewise cubic interpolation
    
    # For efficiency and stability, we use the fact that cubic B-splines
    # have local support. Each basis function N_i(x) is non-zero only
    # for x in [t_i, t_{i+4}] where t are the knot vector.
    
    # For 10 control points with knots at linspace(-3, 3, 10),
    # we extend the knot vector by 3 on each side for proper B-spline definition
    t = np.concatenate([np.full(3, knots[0] - 1.0), knots, np.full(3, knots[-1] + 1.0)])
    
    # Compute B-spline basis functions using recursive definition
    # For cubic B-splines, we can use the Cox-de Boor recursion
    def bspline_basis(x_val, i, p_order, knot_vector):
        """Compute the i-th B-spline basis function of order p_order at x_val."""
        if p_order == 0:
            return 1.0 if knot_vector[i] <= x_val < knot_vector[i + 1] else 0.0
        else:
            denom1 = knot_vector[i + p_order] - knot_vector[i]
            denom2 = knot_vector[i + p_order + 1] - knot_vector[i + 1]
            
            term1 = 0.0
            if denom1 > 0:
                term1 = (x_val - knot_vector[i]) / denom1 * bspline_basis(x_val, i, p_order - 1, knot_vector)
            
            term2 = 0.0
            if denom2 > 0:
                term2 = (knot_vector[i + p_order + 1] - x_val) / denom2 * bspline_basis(x_val, i + 1, p_order - 1, knot_vector)
            
            return term1 + term2
    
    # This recursive approach is too slow for 200+ points.
    # Instead, use a vectorized approach with scipy or a direct formula.
    # Let's use a simpler and faster approach: cubic interpolation via
    # a local basis that approximates B-spline behavior.
    
    # Alternative: use the fact that for uniform knots, we can compute
    # B-spline basis values efficiently.
    
    # For now, use a hybrid approach: combine local cubic basis with
    # global smooth features to ensure good approximation.
    
    # Normalize x to [-1, 1] for numerical stability
    xn = x / 3.0
    
    # Build basis matrix: 10 basis functions
    b = np.ones((10, n), dtype=np.float64)
    
    # 6 local cubic B-spline-like basis functions
    # Centers spread across [-1, 1]
    centers = np.linspace(-1.0, 1.0, 6)
    width = 0.5  # Width of each bump
    
    for i in range(6):
        # Cubic B-spline-like basis: smooth compact support
        diff = (xn - centers[i]) / width
        # Use (1 - diff^2)^3 for |diff| < 1, else 0 (cubic bump)
        b[i, :] = np.where(np.abs(diff) < 1.0, (1.0 - diff**2)**3, 0.0)
    
    # 2 trigonometric basis for periodic components
    b[6, :] = np.sin(np.pi * xn)
    b[7, :] = np.cos(np.pi * xn)
    
    # 2 polynomial basis for global trend
    b[8, :] = xn
    b[9, :] = xn**3
    
    # Linear combination
    result = b @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
