# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

    def compute_sides(centers):
        """
        For each square, compute the maximum possible side length such that it
        remains inside the unit square and does not overlap with any other square.
        
        For square i with center (cx_i, cy_i):
        - Boundary constraint: s_i <= 2 * min(cx_i, 1-cx_i, cy_i, 1-cy_i)
        - Pairwise constraint with square j: 
          If |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2, no overlap.
          This is equivalent to: s_i + s_j <= 2 * max(|cx_i - cx_j|, |cy_i - cy_j|)
          Wait, that's not quite right. The non-overlap condition is:
          |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
          
          For a fixed configuration of centers, we want to find the maximum s_i for each i
          such that all constraints are satisfied. This is a system of linear inequalities.
          
          Actually, let's think differently. For a given set of centers, the maximum side
          length for square i is determined by:
          1. Distance to boundary: min(cx_i, 1-cx_i, cy_i, 1-cy_i) * 2
          2. For each other square j, the constraint is:
             s_i + s_j <= 2 * max(|cx_i - cx_j|, |cy_i - cy_j|)  ... this is not linear.
             
          Let's use the Voronoi idea from the directive. The side length of square i is
          bounded by the distance to its nearest neighbor or boundary. Specifically,
          s_i/2 <= min(
              cx_i, 1-cx_i, cy_i, 1-cy_i,
              min_j (max(|cx_i - cx_j|, |cy_i - cy_j|) - s_j/2)  ... this is circular.
          )
          
          A simpler approach: For a given set of centers, we can compute the maximum
          side length for each square by considering the constraints as:
          s_i <= 2 * min(cx_i, 1-cx_i, cy_i, 1-cy_i)
          And for each pair (i,j): s_i + s_j <= 2 * max(|cx_i - cx_j|, |cy_i - cy_j|)
          
          But this is a coupled system. However, if we assume that the squares are small
          enough that the pairwise constraints are the binding ones, we can iterate.
          
          Alternatively, we can use a simpler bound: 
          s_i <= 2 * min(cx_i, 1-cx_i, cy_i, 1-cy_i, min_j (|cx_i - cx_j|, |cy_i - cy_j|))
          Wait, that's not correct either. If |cx_i - cx_j| = d_x and |cy_i - cy_j| = d_y,
          then the constraint is s_i + s_j <= 2 * max(d_x, d_y). If we assume s_j is small,
          then s_i <= 2 * max(d_x, d_y). But if s_j is large, it's different.
          
          Let's use a different approach: For each square i, the maximum side length is
          limited by the minimum over all other squares j of the "gap" to j, where the gap
          is defined as: if the squares are separated horizontally, the gap is 
          |cx_i - cx_j| - s_j/2 - s_i/2 >= 0, so s_i <= 2 * |cx_i - cx_j| - s_j.
          Similarly for vertical separation.
          
          This is getting complicated. Let's use a simpler heuristic:
          For each square i, compute:
          d_i = min(
              cx_i, 1-cx_i, cy_i, 1-cy_i,
              min_j ( (|cx_i - cx_j| + |cy_i - cy_j|) / 2 )  # Manhattan distance / 2
          )
          Then s_i = 2 * d_i. This is a lower bound on the maximum possible side length,
          and it's smooth (except at the min).
          
          Actually, the L_infinity distance is more appropriate for axis-aligned squares.
          The constraint for non-overlap is:
          |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
          
          If we assume that for each pair, the separation is in one of the two directions,
          we can write:
          s_i + s_j <= 2 * max(|cx_i - cx_j|, |cy_i - cy_j|)
          
          Let's define for each i:
          s_i <= 2 * min(
              cx_i, 1-cx_i, cy_i, 1-cy_i,
              min_j ( max(|cx_i - cx_j|, |cy_i - cy_j|) - s_j/2 )
          )
          This is still coupled.
          
          Let's try a different strategy: Use a penalty-based approach where we define
          an objective that rewards large side lengths but penalizes overlaps.
          
          Objective: maximize sum(s_i) - lambda * sum of overlap penalties
          where the overlap penalty for pair (i,j) is max(0, (s_i + s_j)/2 - max(|cx_i-cx_j|, |cy_i-cy_j|))^2
          and the boundary penalty is max(0, s_i/2 - min(cx_i, 1-cx_i, cy_i, 1-cy_i))^2
          
          But this requires choosing s_i as well. Let's treat both centers and sides as
          optimization variables.
          
          Variables: x = [cx_1, cy_1, s_1, cx_2, cy_2, s_2, ..., cx_10, cy_10, s_10]
          Objective: maximize sum(s_i) - penalty
          Penalty = sum_i max(0, s_i/2 - min(cx_i, 1-cx_i, cy_i, 1-cy_i))^2
                  + sum_{i<j} max(0, (s_i+s_j)/2 - max(|cx_i-cx_j|, |cy_i-cy_j|))^2
          
          We can use scipy.optimize.minimize with a penalty method.
        """
        # This function will be used inside the optimization, so let's not compute
        # sides here. Instead, we'll optimize over centers and sides jointly.
        pass

    def objective_and_grad(x):
        """
        x: array of shape (30,) = [cx_1, cy_1, s_1, cx_2, cy_2, s_2, ..., cx_10, cy_10, s_10]
        Returns: (objective_value, gradient)
        """
        centers = x[0::3].reshape(n, 2)
        sides = x[1::3]
        
        # Ensure sides are positive
        sides = np.maximum(sides, 1e-10)
        
        # Objective: sum of sides
        obj = -np.sum(sides)  # We minimize, so negate
        
        # Penalties
        penalty = 0.0
        grad_penalty = np.zeros_like(x)
        
        # Boundary penalties
        for i in range(n):
            cx = centers[i, 0]
            cy = centers[i, 1]
            s = sides[i]
            
            # Distance to each boundary
            d_left = cx
            d_right = 1.0 - cx
            d_bottom = cy
            d_top = 1.0 - cy
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            if s / 2 > min_d:
                excess = s / 2 - min_d
                penalty += excess ** 2
                # Gradient with respect to s
                grad_penalty[i * 3 + 1] += 2 * excess * 0.5  # d/ds (s/2 - min_d)^2 = 2*(s/2-min_d)*1/2
                # Gradient with respect to cx, cy
                if min_d == d_left:
                    grad_penalty[i * 3 + 0] += 2 * excess * (-0.5)  # d/dcx (cx - s/2)^2 ... wait
                    # Actually, min_d is a function of cx, cy. Let's be more careful.
                    # If min_d = cx, then excess = s/2 - cx, penalty = (s/2 - cx)^2
                    # d/dcx = 2*(s/2-cx)*(-1) = -2*(s/2-cx)
                    # d/ds = 2*(s/2-cx)*(1/2) = (s/2-cx)
                elif min_d == d_right:
                    grad_penalty[i * 3 + 0] += 2 * excess * (0.5)  # d/dcx (1-cx - s/2)^2 ... 
                    # If min_d = 1-cx, excess = s/2 - (1-cx) = s/2 - 1 + cx
                    # penalty = (s/2 - 1 + cx)^2
                    # d/dcx = 2*(s/2-1+cx)*1 = 2*excess
                    # d/ds = 2*(s/2-1+cx)*1/2 = excess
                    grad_penalty[i * 3 + 0] = 2 * excess
                    grad_penalty[i * 3 + 1] = excess
                elif min_d == d_bottom:
                    grad_penalty[i * 3 + 1] += 2 * excess * (-0.5)
                    grad_penalty[i * 3 + 2] = 2 * excess  # wait, index for s is i*3+1
                    # Let me fix the indexing.
                    # x = [cx_1, cy_1, s_1, cx_2, cy_2, s_2, ...]
                    # So for square i: cx is at i*3, cy is at i*3+1, s is at i*3+2
                    pass
                elif min_d == d_top:
                    pass
                # This is getting messy. Let's use a simpler penalty formulation.
        
        # Let's restart with a cleaner penalty formulation.
        # We'll use a smooth approximation of the max function.
        
        return obj, grad_penalty

    # Let's use a simpler approach: optimize only over centers, and compute sides
    # greedily or using a fixed-point iteration.
    
    def compute_sides_from_centers(centers):
        """
        Given centers, compute the maximum side lengths such that all squares
        are non-overlapping and inside the unit square.
        
        We use a simple iterative method: start with small sides, then increase
        each side until it hits a constraint.
        """
        sides = np.full(n, 1e-6)
        
        for iteration in range(100):
            improved = False
            for i in range(n):
                cx, cy = centers[i]
                s_old = sides[i]
                
                # Maximum side based on boundary
                s_boundary = 2 * min(cx, 1-cx, cy, 1-cy)
                
                # Maximum side based on other squares
                s_max = s_boundary
                for j in range(n):
                    if i == j:
                        continue
                    cxj, cyj = centers[j]
                    sj = sides[j]
                    
                    dx = abs(cx - cxj)
                    dy = abs(cy - cyj)
                    
                    # Constraint: s_i + s_j <= 2 * max(dx, dy)
                    # So s_i <= 2 * max(dx, dy) - s_j
                    s_pair = 2 * max(dx, dy) - sj
                    s_max = min(s_max, s_pair)
                
                s_new = max(s_old * 1.01, min(s_max * 0.99, s_boundary * 0.99))
                s_new = max(s_new, 1e-6)
                s_new = min(s_new, s_boundary)
                
                if abs(s_new - s_old) > 1e-10:
                    sides[i] = s_new
                    improved = True
            
            if not improved:
                break
        
        # Final pass: set sides to the maximum possible
        for i in range(n):
            cx, cy = centers[i]
            s_boundary = 2 * min(cx, 1-cx, cy, 1-cy)
            s_max = s_boundary
            for j in range(n):
                if i == j:
                    continue
                cxj, cyj = centers[j]
                sj = sides[j]
                dx = abs(cx - cxj)
                dy = abs(cy - cyj)
                s_pair = 2 * max(dx, dy) - sj
                s_max = min(s_max, s_pair)
            sides[i] = max(min(s_max, s_boundary), 1e-6)
        
        return sides

    def objective(centers):
        """
        Objective function: negative sum of sides (to minimize).
        """
        sides = compute_sides_from_centers(centers)
        return -np.sum(sides)

    # Start from a reasonable initial configuration
    # Use a 3x3 grid plus one extra, or a 2x5 grid
    # Let's try a 2x5 grid first
    k = 2  # 2 columns, 5 rows? Or 5 columns, 2 rows?
    # 10 squares: 2 rows of 5, or 5 rows of 2
    # Let's try 2 rows, 5 columns
    centers_init = np.zeros((n, 2))
    for i in range(n):
        col = i % 5
        row = i // 5
        centers_init[i, 0] = (col + 0.5) / 5
        centers_init[i, 1] = (row + 0.5) / 2
    
    # Optimize
    start_time = time.time()
    deadline = start_time + 580  # Leave 20 seconds margin
    
    best_centers = centers_init.copy()
    best_sides = compute_sides_from_centers(best_centers)
    best_obj = -np.sum(best_sides)
    
    # Use Nelder-Mead for optimization (doesn't require gradient)
    try:
        result = minimize(
            lambda c: objective(c.reshape(n, 2)),
            centers_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_centers = result.x.reshape(n, 2)
            best_sides = compute_sides_from_centers(best_centers)
            best_obj = -np.sum(best_sides)
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            # Try another method: BFGS with numerical gradient
            result2 = minimize(
                lambda c: objective(c.reshape(n, 2)),
                best_centers.flatten(),
                method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-8}
            )
            
            if result2.fun < best_obj:
                best_centers = result2.x.reshape(n, 2)
                best_sides = compute_sides_from_centers(best_centers)
                best_obj = -np.sum(best_sides)
        
    except Exception as e:
        pass
    
    # Ensure validity
    # Clip centers to [0,1] and sides to be positive
    best_centers = np.clip(best_centers, 0, 1)
    best_sides = np.maximum(best_sides, 1e-6)
    
    # Verify and fix if necessary
    for i in range(n):
        cx, cy = best_centers[i]
        s = best_sides[i]
        # Ensure inside unit square
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            # Reduce side
            s = 2 * min(cx, 1-cx, cy, 1-cy)
            s = max(s, 1e-6)
            best_sides[i] = s
    
    # Check for overlaps and fix
    for iter in range(10):
        fixed = False
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = best_centers[i]
                cx_j, cy_j = best_centers[j]
                s_i = best_sides[i]
                s_j = best_sides[j]
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                
                # Check overlap
                if dx < (s_i + s_j)/2 - tol and dy < (s_i + s_j)/2 - tol:
                    # Overlap! Reduce the smaller square's side
                    if s_i < s_j:
                        s_i = max(dx, dy) - s_j/2
                        s_i = max(s_i, 1e-6)
                        best_sides[i] = s_i
                    else:
                        s_j = max(dx, dy) - s_i/2
                        s_j = max(s_j, 1e-6)
                        best_sides[j] = s_j
                    fixed = True
        
        if not fixed:
            break
    
    # Final check
    for i in range(n):
        cx, cy = best_centers[i]
        s = best_sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            s = 2 * min(cx, 1-cx, cy, 1-cy)
            s = max(s, 1e-6)
            best_sides[i] = s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
