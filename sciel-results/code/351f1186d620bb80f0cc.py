"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _validate(circles, max_perim=4.0 + 1e-9):
    """Check if circles are valid: disjoint interiors, bounding box perimeter <= max_perim."""
    if len(circles) < N:
        return False
    # Check bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 0):
        return False
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2 * (w + h)
    if perim > max_perim:
        return False
    # Check disjoint interiors
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                return False
    return True

def _objective_and_constraints(circles, W, H):
    """
    Compute objective (sum of radii) and check constraints.
    We optimize positions and radii given fixed W, H.
    """
    xs = circles[0::3] if circles.ndim == 1 else circles[:, 0]
    ys = circles[1::3] if circles.ndim == 1 else circles[:, 1]
    rs = circles[2::3] if circles.ndim == 1 else circles[:, 2]
    
    # This is for the case where we flatten the array
    # But let's stick to 2D array for clarity in the main function
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = -1.0
    best_circles = None
    
    # Initial guess: grid-like arrangement
    # For 21 circles, a 5x5 grid is 25, so we take 21.
    # Or 3x7 = 21. Let's try a few initial topologies.
    
    # Topology 1: 3 rows of 7
    # Rectangle aspect ratio optimization
    # Perimeter P = 2(W+H) = 4 => W+H=2.
    # Let W = 2 - H.
    # Circles in a 3x7 grid.
    # Approx radius r ~ min(W/7, H/3) / 2 ? No, centers are spaced.
    # If we place centers at (i*W/7 + W/14, j*H/3 + H/6) for i in 0..6, j in 0..2.
    # Max radius is min(W/7, H/3)/2 * something?
    # Actually, for a grid of nx by ny circles in W x H:
    # dx = W/nx, dy = H/ny.
    # r <= min(dx, dy) / 2.
    # Sum r = N * r.
    # We want to maximize N * min(W/nx, H/ny) / 2 subject to W+H=2.
    
    # Let's do a simple grid search over aspect ratios and grid layouts first to get a good starting point.
    # Then use local optimization.
    
    layouts = [
        (3, 7), (7, 3), (4, 6), (6, 4), (5, 5), (2, 11), (11, 2), (1, 21), (21, 1)
    ]
    
    # We will try to optimize each layout
    # State: [x_1, y_1, r_1, ..., x_N, y_N, r_N]
    # Constraints:
    # 1. r_i >= 0
    # 2. x_i - r_i >= 0, x_i + r_i <= W
    # 3. y_i - r_i >= 0, y_i + r_i <= H
    # 4. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Objective: Maximize sum(r_i)
    
    # Since scipy.optimize.minimize minimizes, we minimize -sum(r_i).
    
    # We need to handle the W, H coupling.
    # For a fixed aspect ratio, W and H are fixed.
    # We can iterate over a range of aspect ratios.
    
    # Let's define a function to solve for a specific W, H
    def solve_for_wh(W, H, init_circles=None):
        if W <= 0 or H <= 0:
            return None, -np.inf
        
        # Initialize circles
        if init_circles is None:
            # Simple grid init
            # Find best grid for W, H
            # Just use a random-ish spread or a default 5x5 crop
            # Let's use a 5x5 grid init scaled to W,H
            side = 5
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N: break
                    x = (i + 0.5) * W / side
                    y = (j + 0.5) * H / side
                    r = 0.4 * min(W / side, H / side)
                    cells.append([x, y, r])
            init_circles = np.array(cells[:N])
            
            # Ensure initial validity (shrink radii if needed)
            # Check constraints
            for k in range(10):
                if _validate(init_circles, 4.0): # Wait, _validate checks bounding box perim <= 4.
                    # But here W,H are fixed. The bounding box of the circles might be smaller than W,H.
                    # That's fine. The constraint is that circles are inside [0,W]x[0,H].
                    # My _validate function checks if the *actual* bounding box has perim <= 4.
                    # But in this sub-problem, we are fixing the container W,H such that 2(W+H)=4.
                    # So the container perim is 4.
                    # The circles must be inside.
                    # The objective is sum of radii.
                    # The evaluator measures the raw value (sum of radii) of the returned circles.
                    # The returned circles' bounding box must have perim <= 4.
                    # If we place them in a W x H box with 2(W+H)=4, and they are inside,
                    # their bounding box will have perim <= 4.
                    break
                else:
                    # Shrink radii
                    init_circles[:, 2] *= 0.9
                    
        x0 = init_circles.flatten()
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, W)) # x
            bounds.append((0, H)) # y
            bounds.append((0, min(W, H)/2)) # r (rough bound)
            
        # Constraints
        def constraints(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            
            cons = []
            # Wall constraints
            # x_i - r_i >= 0 => r_i - x_i <= 0
            # x_i + r_i <= W => x_i + r_i - W <= 0
            # y_i - r_i >= 0 => r_i - y_i <= 0
            # y_i + r_i <= H => y_i + r_i - H <= 0
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]}) # x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - (x[i*3] + x[i*3+2])}) # W - x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]}) # y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - (x[i*3+1] + x[i*3+2])}) # H - y_i - r_i >= 0
                
            # Circle-circle constraints
            # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            # This is non-convex. SLSQP might struggle.
            # Alternative: Use a penalty method or trust-constr.
            # Let's try trust-constr or SLSQP with careful initialization.
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda x, i=i, j=j: (x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2 - (x[i*3+2]+x[j*3+2])**2
                    })
            return cons
        
        # Objective
        def objective(x):
            rs = x[2::3]
            return -np.sum(rs)
            
        # Try optimization
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                circles = res.x.reshape(-1, 3)
                # Check validity
                if _validate(circles, 4.0 + 1e-6):
                    score = np.sum(circles[:, 2])
                    return circles, score
        except Exception as e:
            pass
            
        return None, -np.inf

    # Search over aspect ratios
    # W + H = 2
    # W in [0.2, 1.8]
    # Step size 0.1 initially, then refine?
    # Given time budget, let's do a coarse search first.
    
    # Pre-compute some good initial configurations
    # 1. Grid 3x7
    # 2. Grid 4x6 (drop 1)
    # 3. Random
    
    # Let's just run the solver for a few aspect ratios.
    # To save time, we can reuse the best solution from one aspect ratio as init for nearby ones.
    
    aspect_ratios = np.linspace(0.5, 1.5, 11) # W/H ratio? No, W from 0.5 to 1.5 since H=2-W.
    # W in [0.5, 1.5]
    
    prev_circles = None
    
    for W in np.linspace(0.5, 1.5, 21):
        if time.time() > deadline:
            break
        H = 2.0 - W
        circles, score = solve_for_wh(W, H, prev_circles)
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles.copy()
            prev_circles = circles.copy()
            
    # If we found a good one, try to refine it with more iterations
    if best_circles is not None and time.time() < deadline - 10:
        # Get W, H from best_circles bounding box?
        # Or just use the W, H that produced it.
        # Let's re-solve with more iterations using best_circles as init.
        # We need to know which W, H to use.
        # The bounding box of best_circles might be smaller than W, H.
        # Let's compute the tight bounding box.
        xs = best_circles[:, 0]
        ys = best_circles[:, 1]
        rs = best_circles[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        w = max_x - min_x
        h = max_y - min_y
        # We can shift the circles to [0, w] x [0, h]
        # But the perimeter constraint is on the bounding box of the circles.
        # So we can just return the circles as is, as long as perim <= 4.
        # To improve, we can try to expand them.
        
        # Let's try a local expansion.
        # For each circle, try to increase radius slightly, checking constraints.
        # This is slow.
        
        # Instead, let's run the optimizer again with the current best as init, 
        # but allowing W, H to be determined by the bounding box?
        # No, the optimizer fixes W, H.
        
        # Let's just do a final polish using the actual bounding box dimensions.
        # If the bounding box is w x h, we can set W=w, H=h in the optimizer?
        # But then the circles might touch the walls.
        # Let's just stick with the best found.
        
        # One more thing: The evaluator measures sum of radii.
        # If we have extra space, we can expand.
        # Let's try a simple greedy expansion.
        for _ in range(5): # 5 passes
            if time.time() > deadline: break
            improved = False
            for i in range(N):
                if time.time() > deadline: break
                orig_r = best_circles[i, 2]
                # Binary search for max r
                lo, hi = orig_r, orig_r * 1.5
                for _ in range(10):
                    mid = (lo + hi) / 2
                    test_c = best_circles.copy()
                    test_c[i, 2] = mid
                    if _validate(test_c, 4.0):
                        lo = mid
                        improved = True
                    else:
                        hi = mid
                best_circles[i, 2] = lo
            if not improved: break
    
    if best_circles is None:
        # Fallback to a simple grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        # Ensure validity
        while not _validate(best_circles, 4.0):
            best_circles[:, 2] *= 0.9
            
    return best_circles

# EVOLVE-BLOCK-END
