"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # Cubic B-spline basis with 10 control points on domain [-3, 3]
    # Knot vector: uniform spacing, clamped at boundaries
    # 10 control points -> 10 - 4 + 2 = 8 interior knots + 2 boundary knots = 10 knots
    # For cubic B-spline with 10 control points:
    # Number of knots = 10 + 4 = 14 (clamped)
    # Knots: 4 at -3, 8 interior, 4 at 3
    
    # Create clamped knot vector
    num_ctrl = 10
    order = 4  # cubic
    num_knots = num_ctrl + order
    knots = np.zeros(num_knots)
    knots[:order] = -3.0
    knots[-order:] = 3.0
    # Interior knots uniformly spaced
    num_interior = num_knots - 2 * order  # 14 - 8 = 6
    if num_interior > 0:
        interior = np.linspace(-3.0, 3.0, num_interior + 2)[1:-1]
        knots[order:-order] = interior
    
    # Compute cubic B-spline basis functions using de Boor's algorithm
    # For each x_i, find the span and compute the basis values
    basis = np.zeros((n, num_ctrl))
    
    # Find spans for each x
    # x should be in [knots[order-1], knots[num_knots-order-1]] = [-3, 3]
    # Clip x to valid range
    x_clipped = np.clip(x, -3.0, 3.0)
    
    # Find span: for clamped cubic, span index is in [0, num_ctrl-1]
    # Use np.searchsorted on the interior knots
    # The span for x is the index k such that knots[k] <= x < knots[k+1]
    # For clamped B-splines, we can use:
    interior_knots = knots[order:-order]  # 6 interior knots
    # Span = number of interior knots <= x, capped at num_ctrl - 1
    spans = np.searchsorted(interior_knots, x_clipped, side='right')
    spans = np.clip(spans, 0, num_ctrl - 1)
    
    # Compute B-spline basis values using the Cox-de Boor recursion
    # For cubic (degree 3), we need to compute N_{i,0}, N_{i,1}, N_{i,2}, N_{i,3}
    # But we only need the last level for the control point weights
    
    # More efficient: use the explicit formula for cubic B-splines
    # For each point, only 4 basis functions are non-zero
    
    # Let's use the standard algorithm:
    # For span k, the non-zero basis functions are N_{k-3,3}, N_{k-2,3}, N_{k-1,3}, N_{k,3}
    
    for i in range(n):
        k = spans[i]
        xi = x_clipped[i]
        
        # Get the relevant knot intervals
        # N_{j,3}(x) is non-zero only when knots[j] <= x < knots[j+4]
        # For span k, the non-zero functions are j = k-3, k-2, k-1, k
        
        # Compute using de Boor's algorithm
        # Initialize N_{j,0} = 1 if knots[j] <= x < knots[j+1]
        # We need N_{j,3} for j in {k-3, k-2, k-1, k}
        
        # Let's compute all N_{j,d} for d=0,1,2,3 for j in range
        # But more efficiently, use the explicit cubic B-spline formula
        
        # For a uniform clamped B-spline, we can use the standard basis functions
        # But our knots are not uniform (clamped), so let's use recursion
        
        # de Boor's algorithm for a single point
        # N_{j,0}(x) = 1 if knots[j] <= x < knots[j+1], else 0
        # N_{j,d}(x) = (x - knots[j])/(knots[j+d] - knots[j]) * N_{j,d-1}(x) + 
        #              (knots[j+d+1] - x)/(knots[j+d+1] - knots[j+1]) * N_{j+1,d-1}(x)
        
        # We need N_{j,3} for j = k-3, k-2, k-1, k
        # This requires N_{j,d} for j in range [k-3, k] and d in 0..3
        
        # Let's compute a small table
        j_min = max(0, k - 3)
        j_max = min(num_ctrl - 1, k)
        
        # Initialize degree 0
        N_prev = np.zeros(num_ctrl)
        for j in range(num_ctrl):
            if knots[j] <= xi < knots[j+1]:
                N_prev[j] = 1.0
        
        # Recurse up to degree 3
        N_curr = np.zeros(num_ctrl)
        for d in range(1, 4):
            N_curr = np.zeros(num_ctrl)
            for j in range(num_ctrl - d):
                denom1 = knots[j + d] - knots[j]
                denom2 = knots[j + d + 1] - knots[j + 1]
                
                coeff1 = 0.0
                coeff2 = 0.0
                
                if denom1 > 1e-12:
                    coeff1 = (xi - knots[j]) / denom1
                if denom2 > 1e-12:
                    coeff2 = (knots[j + d + 1] - xi) / denom2
                
                N_curr[j] = coeff1 * N_prev[j] + coeff2 * N_prev[j + 1]
            
            N_prev = N_curr
        
        # N_prev now contains N_{j,3} for all j
        # Assign to basis row
        for j in range(j_min, j_max + 1):
            basis[i, j] = N_prev[j]
    
    # Linear combination
    result = basis @ params
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
