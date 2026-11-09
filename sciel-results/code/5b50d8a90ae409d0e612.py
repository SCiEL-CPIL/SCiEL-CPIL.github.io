"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise cubic B-spline with 9 fixed internal knots in [-2, 2].
    params[0..8] are the 9 control points (coefficients of the cubic B-spline basis).
    params[9] is an additive constant (offset) to allow vertical shift.
    
    The knots are placed at equally spaced intervals in [-2, 2]:
    Boundary knots: -2, -1, 0, 1, 2 (with multiplicity 4 for clamped B-spline)
    Internal knots: -1.5, -0.5, 0.5, 1.5 (4 internal knots)
    
    For a cubic B-spline (degree 3) with 9 basis functions, we need:
    - 4 boundary knots on each side (total 8 boundary knots)
    - 4 internal knots
    Total knots: 12, giving 9 basis functions (12 - 3 - 1 + 1 = 9)
    
    Wait, let me recalculate. For clamped cubic B-spline:
    - We have knots: k_0, k_1, ..., k_{n+4} where n is the number of basis functions
    - For n=9 basis functions, we need 9+4-1 = 12 knots
    - Clamped: first 4 knots = -2, last 4 knots = 2
    - Internal knots: 12 - 8 = 4 knots at -1.5, -0.5, 0.5, 1.5
    
    So the knot vector is: [-2, -2, -2, -2, -1.5, -0.5, 0.5, 1.5, 2, 2, 2, 2]
    This gives 9 cubic B-spline basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = x.shape[0]
    
    # Fixed knot vector for clamped cubic B-spline with 9 basis functions
    # Knots: -2 (x4), -1.5, -0.5, 0.5, 1.5, 2 (x4)
    knots = np.array([-2.0, -2.0, -2.0, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 2.0, 2.0, 2.0])
    
    # Compute cubic B-spline basis functions using de Boor's algorithm / recursive Cox-de Boor
    # For efficiency, we compute all 9 basis values at each x point
    
    # Number of basis functions
    n_basis = 9
    
    # Initialize basis matrix: shape (n_points, n_basis)
    basis = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # Cox-de Boor recursion for cubic B-spline (degree 3)
    # B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1] else 0
    
    # Degree 0
    for i in range(len(knots) - 1):
        mask = (x >= knots[i]) & (x < knots[i + 1])
        if i < n_basis:
            basis[:, i] = mask.astype(np.float64)
    
    # Degree 1
    for i in range(len(knots) - 2):
        denom1 = knots[i + 1] - knots[i]
        denom2 = knots[i + 2] - knots[i + 1]
        if denom1 > 0 and denom2 > 0:
            mask1 = (x >= knots[i]) & (x < knots[i + 1])
            mask2 = (x >= knots[i + 1]) & (x < knots[i + 2])
            if i < n_basis:
                basis[:, i] = (denom2 / (denom1 + denom2)) * (x - knots[i]) / denom1 * mask1.astype(np.float64) \
                             + (denom1 / (denom1 + denom2)) * (knots[i + 2] - x) / denom2 * mask2.astype(np.float64)
    
    # Actually, let me use a cleaner recursive approach. Let me redo this properly.
    
    # Reset basis
    basis = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # Cox-de Boor recursive algorithm
    # B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
    for i in range(n_basis):
        basis[:, i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(np.float64)
    
    # Recurse for degree 1, 2, 3
    for d in range(1, 4):
        for i in range(n_basis - d, -1, -1):
            # B_{i,d}(x) = (x - knots[i]) / (knots[i+d] - knots[i]) * B_{i,d-1}(x) 
            #            + (knots[i+d+1] - x) / (knots[i+d+1] - knots[i+1]) * B_{i+1,d-1}(x)
            
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            
            if denom1 > 1e-12:
                term1 = (x - knots[i]) / denom1 * basis[:, i]
            else:
                term1 = np.zeros(n_points, dtype=np.float64)
            
            if denom2 > 1e-12:
                term2 = (knots[i + d + 1] - x) / denom2 * basis[:, i + 1]
            else:
                term2 = np.zeros(n_points, dtype=np.float64)
            
            basis[:, i] = term1 + term2
    
    # The first n_basis columns of basis now contain the cubic B-spline values
    # Compute the spline value: sum of control_point[i] * basis[i](x) + offset
    control_points = params[:9]
    offset = params[9]
    
    result = basis @ control_points + offset
    
    return result
# EVOLVE-BLOCK-END
