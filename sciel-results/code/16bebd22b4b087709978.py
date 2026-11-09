# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    # Variable vector: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
    # x = [cx_i, cy_i, s_i] for i in 0..n-1
    # Total dimension: 3*n = 30
    
    def unpack(x):
        """Unpack variable vector into centers and sides."""
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides
    
    def objective(x):
        """Negative sum of sides (to minimize)."""
        _, sides = unpack(x)
        return -np.sum(sides)
    
    def constraints(x):
        """
        Constraints:
        1. Containment: for each square i,
           cx_i - s_i/2 >= 0, cx_i + s_i/2 <= 1
           cy_i - s_i/2 >= 0, cy_i + s_i/2 <= 1
        2. Non-overlap: for each pair (i,j), i<j,
           |cx_i - cx_j| >= (s_i + s_j)/2  OR  |cy_i - cy_j| >= (s_i + s_j)/2
           
        For non-overlap, we use the fact that squares don't overlap if they are
        separated in at least one dimension. We encode this using binary variables
        or by trying all 4 combinations of separation directions.
        
        Actually, a cleaner approach: two axis-aligned squares don't overlap if
        one is to the left of the other, or one is above the other.
        
        Square i and j don't overlap iff:
          cx_i + s_i/2 <= cx_j - s_j/2  (i left of j)
          OR cx_j + s_j/2 <= cx_i - s_i/2  (j left of i)
          OR cy_i + s_i/2 <= cy_j - s_j/2  (i below j)
          OR cy_j + s_j/2 <= cy_i - s_i/2  (j below i)
        
        This is a disjunctive constraint. We can handle it by introducing
        binary variables, but that makes it a mixed-integer problem.
        
        Alternative: Use the fact that for axis-aligned squares, non-overlap
        can be checked by:
          (cx_i - s_i/2 >= cx_j + s_j/2) OR (cx_j - s_j/2 >= cx_i + s_i/2) OR
          (cy_i - s_i/2 >= cy_j + s_j/2) OR (cy_j - s_j/2 >= cy_i + s_i/2)
        
        For optimization, we can use a penalty approach or try multiple
        initializations with different "separation patterns".
        
        Let me use a different approach: parameterize using the bottom-left
        corners and sides, and use a smooth approximation or try all 2^(4*C(n,2))
        separation patterns (too many for n=10).
        
        Better: Use a penalty method with scipy's SLSQP.
        """
        centers, sides = unpack(x)
        cons = []
        
        # Containment constraints
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # cx - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]/2.0})
            # cx + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]/2.0})
            # cy - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]/2.0})
            # cy + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]/2.0})
            # s > 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*n+i] - 1e-10})
        
        # Non-overlap constraints using smooth approximation
        # For each pair (i,j), enforce separation in at least one dimension
        # We use a penalty-based approach: minimize overlap area
        # But for SLSQP, let's try to enforce that the minimum separation is >= 0
        
        # Actually, let's use a different strategy: try multiple random
        # initializations and use a penalty function for overlaps.
        
        return cons
    
    # Better approach: Use a penalty method with Nelder-Mead or SLSQP
    # Let me implement a proper optimization with multiple restarts
    
    def compute_overlap_penalty(x, softness=100.0):
        """Compute penalty for overlaps and boundary violations."""
        centers, sides = unpack(x)
        penalty = 0.0
        
        # Boundary violations
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left boundary
            if cx - s/2 < 0:
                penalty += softness * (0 - (cx - s/2))**2
            # Right boundary
            if cx + s/2 > 1:
                penalty += softness * ((cx + s/2) - 1)**2
            # Bottom boundary
            if cy - s/2 < 0:
                penalty += softness * (0 - (cy - s/2))**2
            # Top boundary
            if cy + s/2 > 1:
                penalty += softness * ((cy + s/2) - 1)**2
            # Side must be positive
            if s < 1e-10:
                penalty += softness * (1e-10 - s)**2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i, s_j = sides[i], sides[j]
                
                # Check separation in x
                sep_x = max(cx_i + s_i/2 - cx_j + s_j/2,
                           cx_j + s_j/2 - cx_i + s_i/2, 0)
                # Check separation in y
                sep_y = max(cy_i + s_i/2 - cy_j + s_j/2,
                           cy_j + s_j/2 - cy_i + s_i/2, 0)
                
                # If both separations are negative, there's overlap
                # The amount of overlap in x direction
                overlap_x = max(0, (s_i + s_j)/2 - abs(cx_i - cx_j))
                overlap_y = max(0, (s_i + s_j)/2 - abs(cy_i - cy_j))
                
                # Penalty if both overlaps are positive
                if overlap_x > 0 and overlap_y > 0:
                    penalty += softness * overlap_x * overlap_y
        
        return penalty
    
    def penalized_objective(x, softness=100.0):
        """Objective with penalty for constraint violations."""
        _, sides = unpack(x)
        return -np.sum(sides) + compute_overlap_penalty(x, softness)
    
    # Start with a good initial guess (grid packing)
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    sides_init = np.full(n, side, dtype=np.float64)
    
    # Pack into variable vector
    x0 = np.zeros(3*n)
    x0[0::2] = cx  # cx values
    x0[1::2] = cy  # cy values
    x0[2*n:] = sides_init
    
    # Try multiple random initializations and optimize
    best_x = x0.copy()
    best_obj = penalized_objective(x0, 100.0)
    
    # Try the grid initialization first
    res = minimize(penalized_objective, x0, args=(100.0,), method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    if res.fun < best_obj:
        best_obj = res.fun
        best_x = res.x
    
    # Try multiple random initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        # Try different grid patterns
        k_try = np.random.choice([2, 3, 4])
        side_try = 1.0 / k_try
        idx_try = np.arange(n)
        cols_try = idx_try % k_try
        rows_try = idx_try // k_try
        
        cx_try = (cols_try + 0.5) * side_try + np.random.randn(n) * 0.01
        cy_try = (rows_try + 0.5) * side_try + np.random.randn(n) * 0.01
        sides_try = np.full(n, side_try, dtype=np.float64) * (1 + np.random.randn(n) * 0.1)
        sides_try = np.clip(sides_try, 0.05, 1.0)
        
        # Clamp centers to keep squares roughly inside
        cx_try = np.clip(cx_try, 0.05, 0.95)
        cy_try = np.clip(cy_try, 0.05, 0.95)
        
        x_try = np.zeros(3*n)
        x_try[0::2] = cx_try
        x_try[1::2] = cy_try
        x_try[2*n:] = sides_try
        
        res_try = minimize(penalized_objective, x_try, args=(100.0,), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res_try.fun < best_obj:
            best_obj = res_try.fun
            best_x = res_try.x
    
    # Extract best solution
    centers, sides = unpack(best_x)
    
    # Clean up: ensure all values are finite and valid
    sides = np.maximum(sides, 1e-10)
    centers = np.clip(centers, 0, 1)
    
    # Final validation and cleanup
    for i in range(n):
        s = sides[i]
        centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
        centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
    
    centers = centers.astype(np.float64)
    sides = sides.astype(np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
