import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares 
    packed inside the unit square [0, 1]^2.
    """
    n = 10
    
    # Initial heuristic layout
    # We try to pack squares of varying sizes.
    # A known good configuration involves a large square, medium squares, and small fillers.
    
    # Let's define initial centers and sides.
    # Layout:
    # Bottom-left: 1 large square (s=0.5)
    # Bottom-right: 1 large square (s=0.5)
    # Top-left: 2 medium squares (s=0.25) side by side?
    # Top-right: 2 medium squares (s=0.25) side by side?
    # This leaves a gap. 
    # Let's use a more organic initial guess and let the optimizer work.
    
    # Initial guess:
    # 4 squares of size 0.4 in corners? No, 0.4*2 = 0.8, leaves 0.2 gap.
    # Let's try:
    # 1 square s=0.5 at (0.25, 0.25)
    # 1 square s=0.5 at (0.75, 0.25)
    # 2 squares s=0.25 at (0.125, 0.625) and (0.375, 0.625)
    # 2 squares s=0.25 at (0.625, 0.625) and (0.875, 0.625)
    # Remaining 4 squares must fit in the top strip y in [0.75, 1.0]?
    # The top of the 0.25 squares is 0.625 + 0.125 = 0.75.
    # So we have a strip of height 0.25 from y=0.75 to y=1.0.
    # Width is 1.0.
    # We can fit 4 squares of size 0.25 in this strip? 4 * 0.25 = 1.0 width.
    # Yes.
    
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.875],
        [0.875, 0.875]
    ])
    sides_init = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    
    # This layout is valid. Sum = 1.0 + 1.0 + 1.0 = 3.0.
    # Wait, the baseline grid (k=4) gave sum = 10 * 0.25 = 2.5.
    # My heuristic gives 3.0. This is already better than 1.0 score (which was 2.5/2.5=1.0? No, score is value/seed. Seed value was 2.5. So 3.0/2.5 = 1.2).
    # Can we do better?
    # The optimal for 10 is around 3.4.
    
    # Let's optimize this layout.
    
    def penalty_and_objective(x):
        # x contains 20 variables: cx0, cy0, s0, cx1, cy1, s1, ...
        # s_i > 0
        # 0 <= cx_i - s_i/2 and cx_i + s_i/2 <= 1
        # 0 <= cy_i - s_i/2 and cy_i + s_i/2 <= 1
        # No overlaps: |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
        
        centers = x[0::3].reshape(-1, 2) # This is wrong, x is flat [cx, cy, s, cx, cy, s ...]
        # Let's unpack properly
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Ensure positive sides
        s = np.maximum(s, 1e-6)
        
        # Boundary penalties
        penalty = 0.0
        tol = 1e-7
        for i in range(n):
            if cx[i] - s[i]/2 < -tol:
                penalty += 1000 * (-tol - (cx[i] - s[i]/2))
            if cx[i] + s[i]/2 > 1 + tol:
                penalty += 1000 * ((cx[i] + s[i]/2) - (1 + tol))
            if cy[i] - s[i]/2 < -tol:
                penalty += 1000 * (-tol - (cy[i] - s[i]/2))
            if cy[i] + s[i]/2 > 1 + tol:
                penalty += 1000 * ((cy[i] + s[i]/2) - (1 + tol))
                
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_needed = (s[i] + s[j]) / 2
                # If both dx < dist_needed and dy < dist_needed, they overlap
                # We want to push them apart.
                # A smooth penalty: max(0, dist_needed - dx)^2 + max(0, dist_needed - dy)^2 ?
                # Actually, if they overlap, we want to increase dx or dy.
                # Penalty = max(0, dist_needed - dx) * max(0, dist_needed - dy) ? No.
                # If they overlap, both conditions are true.
                # Let's use: if dx < dist_needed and dy < dist_needed:
                #    penalty += (dist_needed - dx) * (dist_needed - dy) * 100
                # This encourages separation.
                
                if dx < dist_needed and dy < dist_needed:
                    # Overlap area is roughly (dist_needed - dx) * (dist_needed - dy)
                    # We penalize the overlap
                    penalty += 100 * (dist_needed - dx) * (dist_needed - dy)
                    
        # Objective: minimize negative sum of sides + penalty
        return -(np.sum(s)) + penalty

    # Flatten initial state
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = sides_init[i]
        
    # Start time
    start_time = time.time()
    timeout = 590 # Leave 10s margin
    
    # Use Nelder-Mead
    # We can run multiple restarts or just one long run.
    # Given the landscape, Nelder-Mead can get stuck.
    # Let's try a few iterations.
    
    best_x = x0
    best_val = penalty_and_objective(x0)
    
    # We will run optimization until time runs out.
    # To allow for multiple restarts or different initializations if needed,
    # but for now, let's stick to one strong run with the heuristic start.
    
    # Actually, Nelder-Mead on 30 dimensions might be slow or inefficient.
    # Let's try to reduce dimensions or use a different method.
    # However, scipy is available.
    
    # Let's try a simple local search with random perturbations if Nelder-Mead is too slow.
    # But Nelder-Mead is usually fast enough for 30 dims for a few seconds.
    
    result = None
    try:
        # Set maxiter to a high number, but we will interrupt if time runs out
        # scipy minimize doesn't have a time limit directly, so we run in a loop?
        # Or just set maxiter to something reasonable like 10000.
        # 10000 iterations of Nelder-Mead on 30 dims is likely < 10s.
        
        # Let's run it.
        res = minimize(penalty_and_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() - start_time < timeout:
        # Try a second run with a slightly different start?
        # Or just return the best found.
        pass
        
    # Unpack best_x
    cx = best_x[0::3]
    cy = best_x[1::3]
    s = best_x[2::3]
    
    # Final validation and cleanup
    # Ensure sides are positive
    s = np.maximum(s, 1e-6)
    
    # Clamp centers to ensure boundaries are respected strictly?
    # The penalty should have handled it, but let's be safe.
    for i in range(n):
        # If a square is out of bounds, push it in
        # This is a bit crude but ensures validity if penalty failed slightly
        if cx[i] - s[i]/2 < 0:
            cx[i] = s[i]/2
        if cx[i] + s[i]/2 > 1:
            cx[i] = 1 - s[i]/2
        if cy[i] - s[i]/2 < 0:
            cy[i] = s[i]/2
        if cy[i] + s[i]/2 > 1:
            cy[i] = 1 - s[i]/2
            
    centers = np.stack([cx, cy], axis=1)
    sides = s
    
    # Verify validity one last time
    # If invalid, fall back to the initial heuristic which is known valid
    valid = True
    tol = 1e-7
    for i in range(n):
        if not (centers[i, 0] - sides[i]/2 >= -tol and centers[i, 0] + sides[i]/2 <= 1 + tol):
            valid = False
        if not (centers[i, 1] - sides[i]/2 >= -tol and centers[i, 1] + sides[i]/2 <= 1 + tol):
            valid = False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j])/2 - tol
            if dx < t and dy < t:
                valid = False
                
    if not valid:
        # Fallback to initial heuristic
        centers = centers_init
        sides = sides_init

    return centers, sides
