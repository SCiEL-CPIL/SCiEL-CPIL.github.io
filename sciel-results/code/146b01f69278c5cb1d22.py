"""
Strategy: Linear Spline Basis (Piecewise Linear Hat Functions).
Instead of global polynomials which can suffer from ill-conditioning and oscillation (Runge's phenomenon) on [-3, 3],
we use a local basis of linear "hat" functions. This provides a stable, sparse representation that BFGS can optimize
effectively. With 10 parameters, we define 10 hat functions centered at specific nodes across the domain.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear spline basis (hat functions).
    
    We define 10 basis functions corresponding to 10 parameters.
    The basis functions are piecewise linear 'hats' centered at nodes distributed
    across the domain [-3, 3]. This local basis is well-conditioned for BFGS
    and avoids the global oscillation issues of high-degree polynomials.
    
    Nodes are chosen to cover the domain. To ensure coverage of the edges,
    we place nodes slightly inside the boundaries or use a grid that spans the range.
    With 10 params, we can define 10 centers.
    Let's use centers: -2.7, -2.0, -1.3, -0.6, 0.1, 0.8, 1.5, 2.2, 2.9
    Wait, 10 params. Let's use 10 centers evenly spaced.
    Centers: -2.7, -1.95, -1.2, -0.45, 0.3, 1.05, 1.8, 2.55 ... 
    Actually, simpler: define a grid of 10 points in [-2.8, 2.8] to ensure hats don't cut off too early.
    Or even simpler: use a fixed set of 10 centers.
    
    Let's use centers: c_i = -3 + 6*(i+0.5)/10 for i in 0..9?
    c_0 = -3 + 0.3 = -2.7
    c_1 = -3 + 0.9 = -2.1
    c_2 = -3 + 1.5 = -1.5
    c_3 = -3 + 2.1 = -0.9
    c_4 = -3 + 2.7 = -0.3
    c_5 = -3 + 3.3 = 0.3
    c_6 = -3 + 3.9 = 0.9
    c_7 = -3 + 4.5 = 1.5
    c_8 = -3 + 5.1 = 2.1
    c_9 = -3 + 5.7 = 2.7
    
    The width of each hat is determined by the spacing to neighbors.
    Spacing h = 0.6.
    Hat i is non-zero in [c_i - h, c_i + h] = [c_{i-1}, c_{i+1}].
    Specifically, for interior points, it's 0 outside [c_{i-1}, c_{i+1}].
    For boundary points, we extend the support to the domain edge.
    
    This basis is sparse: each x activates at most 2 or 3 basis functions.
    This is extremely fast and stable for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 centers
    # We want them to cover [-3, 3] well.
    # Using the spacing derived above: h = 0.6
    centers = np.array([-2.7, -2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1, 2.7])
    
    # Precompute supports
    # For each center i, the support is [left_i, right_i]
    # left_0 = -3.0 (domain edge), right_0 = c_1 = -2.1
    # left_1 = c_0 = -2.7, right_1 = c_2 = -1.5
    # ...
    # left_9 = c_8 = 2.1, right_9 = 3.0 (domain edge)
    
    lefts = np.zeros(10)
    rights = np.zeros(10)
    
    lefts[0] = -3.0
    rights[0] = centers[1]
    
    for i in range(1, 9):
        lefts[i] = centers[i-1]
        rights[i] = centers[i+1]
        
    lefts[9] = centers[8]
    rights[9] = 3.0
    
    # Initialize output
    y = np.zeros_like(x)
    
    # For each basis function, compute contribution
    for i in range(10):
        c = centers[i]
        l = lefts[i]
        r = rights[i]
        
        # Mask of points within support
        mask = (x >= l) & (x <= r)
        
        if np.any(mask):
            x_masked = x[mask]
            
            # Linear hat shape
            # From l to c: slope = 1 / (c - l)
            # From c to r: slope = -1 / (r - c)
            
            # Calculate values
            # val = max(0, 1 - |x - c| / half_width) is for symmetric, but supports are not always symmetric
            # Better: piecewise linear interpolation of (l,0), (c,1), (r,0)
            
            # Left part: l <= x <= c
            mask_left = x_masked <= c
            # Right part: c < x <= r
            mask_right = ~mask_left
            
            vals = np.zeros_like(x_masked)
            
            if np.any(mask_left):
                # (x - l) / (c - l)
                denom_l = c - l
                if denom_l > 1e-12:
                    vals[mask_left] = (x_masked[mask_left] - l) / denom_l
            
            if np.any(mask_right):
                # (r - x) / (r - c)
                denom_r = r - c
                if denom_r > 1e-12:
                    vals[mask_right] = (r - x_masked[mask_right]) / denom_r
            
            y[mask] += params[i] * vals
            
    return y
# EVOLVE-BLOCK-END
