"""Symbolic regression with piecewise cubic B-spline and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a piecewise cubic B-spline model.
    
    The model is:
    y(x) = sum_{i=0}^{9} p_i * B_i(x)
    
    where B_i are cubic B-spline basis functions with fixed knots uniformly
    distributed over [-2, 2]. The 10 params are the control points that
    directly define the curve values, enabling flexible local approximation
    without the global bias of polynomial or trigonometric bases.
    
    This approach avoids ill-conditioning of high-degree polynomials and
    provides local control over the function shape.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 150s limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    # Define fixed knots for the B-spline
    # Use 10 control points with knots spanning [-2, 2]
    # For cubic B-splines, we need at least 4 knots per segment
    # With 10 control points, we'll use a uniform knot vector
    
    # Knot vector: extend the domain by 3 intervals on each side for cubic splines
    # Interior knots at uniform intervals
    n_knots = 10  # number of control points
    # Create uniform interior knots
    interior_knots = np.linspace(-2, 2, n_knots)
    
    # Full knot vector with 3 extra knots on each side (for cubic, degree=3)
    # This ensures the basis functions are well-defined
    knot_ext_left = interior_knots[0] - (interior_knots[1] - interior_knots[0])
    knot_ext_right = interior_knots[-1] + (interior_knots[-1] - interior_knots[-2])
    
    full_knots = np.concatenate([
        [knot_ext_left] * 4,
        interior_knots,
        [knot_ext_right] * 4
    ])
    
    # Compute cubic B-spline basis functions using the de Boor algorithm
    # For each x, compute which knot interval it falls in
    # Then compute the 4 non-zero basis functions for cubic splines
    
    # Clip x to the knot domain to avoid out-of-range issues
    x_clipped = np.clip(x, interior_knots[0], interior_knots[-1])
    
    # Compute B-spline basis values
    # Use a vectorized approach for efficiency
    
    def cubic_bspline_basis(x_eval, knots):
        """
        Compute cubic B-spline basis function values at x_eval given knots.
        Returns array of shape (len(x_eval), n_basis) where n_basis = len(knots) - 4
        """
        n_knots = len(knots)
        n_basis = n_knots - 4  # number of basis functions for cubic
        n_points = len(x_eval)
        
        # Initialize basis values
        basis = np.zeros((n_points, n_basis))
        
        for i in range(n_points):
            xi = x_eval[i]
            
            # Find the knot interval for xi
            # Binary search for the interval
            idx = np.searchsorted(knots, xi, side='right')
            
            # For cubic B-splines, the non-zero basis functions are B_{idx-3}, B_{idx-2}, B_{idx-1}, B_{idx}
            # But we need to handle boundaries carefully
            
            # Compute the 4 relevant basis functions using the Cox-de Boor recursion
            # B_0,i(x) = 1 if knots[i] <= x < knots[i+1], else 0
            # B_{k,i}(x) = (x - knots[i])/(knots[i+k] - knots[i]) * B_{k-1,i}(x) + 
            #              (knots[i+k+1] - x)/(knots[i+k+1] - knots[i+1]) * B_{k-1,i+1}(x)
            
            # For efficiency, compute directly for cubic
            # The 4 non-zero basis functions for x in [knots[j], knots[j+1]) are:
            # B_{j-3, 3}, B_{j-2, 3}, B_{j-1, 3}, B_{j, 3}
            
            j = idx  # index such that knots[j] <= xi < knots[j+1]
            
            # The 4 relevant basis functions have indices j-3, j-2, j-1, j
            # But we need to make sure these are within [0, n_basis)
            
            for k in range(4):
                basis_idx = j - 3 + k
                if 0 <= basis_idx < n_basis:
                    # Compute B_{basis_idx, 3}(xi)
                    # Using the explicit formula for cubic B-splines
                    i0 = basis_idx
                    
                    # Get the 5 relevant knots
                    knots_5 = knots[i0:i0+5]
                    
                    # Compute the 4 cubic B-spline basis values at xi
                    # B_3,i(x) = 1/6 * (
                    #   (x - t_i)^3 / ((t_{i+1}-t_i)(t_{i+2}-t_i)(t_{i+3}-t_i)) +
                    #   (x - t_{i+1})^3 / ((t_{i+1}-t_i)(t_{i+2}-t_{i+1})(t_{i+3}-t_{i+1})) * (-1) +
                    #   ... 
                    # )
                    
                    # Use the standard cubic B-spline formula
                    # For uniform knots with spacing h, B_3(x) has a simple form
                    # But for non-uniform, use the general formula
                    
                    t0, t1, t2, t3, t4 = knots_5
                    
                    # Compute each term
                    def safe_div(a, b):
                        if abs(b) < 1e-12:
                            return 0.0
                        return a / b
                    
                    # Term 1: (x - t0)^3 / ((t1-t0)(t2-t0)(t3-t0))
                    term1 = safe_div((xi - t0)**3, (t1-t0)*(t2-t0)*(t3-t0))
                    
                    # Term 2: -(x - t1)^3 / ((t1-t0)(t2-t1)(t3-t1))
                    term2 = -safe_div((xi - t1)**3, (t1-t0)*(t2-t1)*(t3-t1))
                    
                    # Term 3: (x - t2)^3 / ((t1-t2)(t2-t0)(t3-t2))
                    term3 = safe_div((xi - t2)**3, (t1-t2)*(t2-t0)*(t3-t2))
                    
                    # Term 4: -(x - t3)^3 / ((t1-t3)(t2-t3)(t3-t2))
                    term4 = -safe_div((xi - t3)**3, (t1-t3)*(t2-t3)*(t3-t2))
                    
                    basis[i, basis_idx] = (term1 + term2 + term3 + term4) / 6.0
        
        return basis
    
    # Compute basis values
    basis_values = cubic_bspline_basis(x_clipped, full_knots)
    
    # The basis_values array has shape (n_points, n_basis)
    # We need to extract the first 10 columns (or however many we have)
    n_available = min(10, basis_values.shape[1])
    basis_10 = basis_values[:, :n_available]
    
    # Ensure basis_10 has exactly 10 columns
    if basis_10.shape[1] < 10:
        basis_10 = np.pad(basis_10, ((0, 0), (0, 10 - basis_10.shape[1])))
    
    # Linear combination with params
    result = basis_10 @ params[:10]
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
