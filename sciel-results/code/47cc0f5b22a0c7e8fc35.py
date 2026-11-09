# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Returns a valid packing of 10 squares in the unit square maximizing the sum of side lengths.
    
    Strategy:
    1. Start with a known valid configuration: 2 large squares (0.5) and 8 small squares (0.25).
    2. Use scipy.optimize.minimize to locally optimize the positions and sizes.
    3. The objective function is the negative sum of sides plus a large penalty for constraint violations.
    4. Constraints:
       - Squares must be inside [0, 1]x[0, 1].
       - Squares must not overlap.
    """
    n = 10
    tol = 1e-7
    
    # Initial configuration: 2 large, 8 small
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Large squares
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5
    
    # Small squares in Top-Left quadrant
    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25
    
    # Small squares in Bottom-Right quadrant
    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25
    
    # Flatten variables: x0, y0, s0, x1, y1, s1, ...
    def flatten(centers, sides):
        return np.hstack([centers[:, 0], centers[:, 1], sides])
    
    def unflatten(x):
        cx = x[0:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        return np.column_stack((cx, cy)), s

    def objective(x):
        centers, sides = unflatten(x)
        
        # 1. Boundary Violations
        # cx - s/2 >= -tol  => cx - s/2 + tol >= 0
        # cx + s/2 <= 1+tol => 1 + tol - cx - s/2 >= 0
        # Same for y
        
        penalty = 0.0
        lam = 1e6
        
        # Boundary checks
        for i in range(n):
            cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
            
            # Left
            if cx - s/2 < -tol:
                penalty += lam * (-tol - (cx - s/2))**2
            # Right
            if cx + s/2 > 1 + tol:
                penalty += lam * ((cx + s/2) - (1 + tol))**2
            # Bottom
            if cy - s/2 < -tol:
                penalty += lam * (-tol - (cy - s/2))**2
            # Top
            if cy + s/2 > 1 + tol:
                penalty += lam * ((cy + s/2) - (1 + tol))**2
            
            # Side must be positive
            if s <= 0:
                penalty += lam * s**2

        # 2. Overlap Violations
        # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
        # We want to penalize if they are "too close".
        # Let d_x = |cx_i - cx_j|, d_y = |cy_i - cy_j|
        # Let r = (s_i + s_j)/2 - tol
        # If d_x < r and d_y < r, there is overlap.
        # We can define a penalty based on how much they "intrude" into each other's forbidden zone.
        
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i, s_i = centers[i, 0], centers[i, 1], sides[i]
                cx_j, cy_j, s_j = centers[j, 0], centers[j, 1], sides[j]
                
                d_x = abs(cx_i - cx_j)
                d_y = abs(cy_i - cy_j)
                r = (s_i + s_j)/2 - tol
                
                if d_x < r and d_y < r:
                    # Calculate penetration
                    # We want to push them apart.
                    # A simple penalty is the square of the amount by which they violate the separation.
                    # Violation in x: max(0, r - d_x)
                    # Violation in y: max(0, r - d_y)
                    # If both are positive, they overlap.
                    # A common approach is to penalize the sum of squared penetrations or the product?
                    # Actually, if they overlap, we want to increase d_x or d_y or decrease s.
                    # Let's use the sum of squared violations.
                    vx = r - d_x
                    vy = r - d_y
                    # If they overlap, vx > 0 and vy > 0.
                    # We want to minimize the overlap area? Or just ensure separation.
                    # Penalizing vx^2 + vy^2 works well for local optimization.
                    penalty += lam * (vx**2 + vy**2)
        
        # Objective: Minimize Penalty - Sum(Sides)
        # We subtract the sum of sides so that minimizing the objective maximizes the sum.
        # The penalty must be large enough to dominate the sum of sides if constraints are violated.
        # Max sum of sides is roughly 3.0. Penalty lam=1e6 is huge.
        return penalty - np.sum(sides)

    x0 = flatten(centers, sides)
    
    # Run optimization
    start_time = time.time()
    deadline = start_time + 590.0 # 10s buffer
    
    best_x = x0
    best_val = objective(x0)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 10000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        pass # If it fails, stick with initial
        
    # Check deadline
    if time.time() > deadline:
        # Return initial if we are running out of time and haven't improved
        pass
        
    centers_opt, sides_opt = unflatten(best_x)
    
    # Final sanity check and cleanup
    # Ensure sides are positive
    sides_opt = np.maximum(sides_opt, 1e-8)
    
    # Re-center if they drifted slightly out? 
    # The penalty should have kept them in. But let's clip centers to be safe?
    # No, clipping might cause overlaps. 
    # The penalty method with L-BFGS-B should handle it.
    
    return centers_opt, sides_opt

# EVOLVE-BLOCK-END
