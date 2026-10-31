"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s (or 150s) kill
    
    # We want to pack N circles in a rectangle of perimeter 4.
    # Perimeter P = 2(w+h) = 4 => w+h = 2.
    # Let aspect ratio r = w/h. Then w = 2*r/(1+r), h = 2/(1+r).
    # Maximize sum of radii.
    
    # Initial guess: square-ish grid
    # For 21 circles, a 5x5 grid with 4 removed? Or 6x4?
    # Let's try a few initial aspect ratios and grid layouts.
    
    best_val = 0.0
    best_circles = None
    
    # Helper to evaluate a configuration
    def evaluate(circles, w, h):
        # circles: (n, 3) array of [x, y, r]
        # Check constraints
        if len(circles) != N:
            return 0.0
        
        # Check bounds
        for x, y, r in circles:
            if x < -r or x > w + r or y < -r or y > h + r:
                return 0.0
            if r <= 0:
                return 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2 - 1e-9:
                    return 0.0
        
        return np.sum(circles[:, 2])
    
    # Strategy:
    # 1. Try a few fixed aspect ratios.
    # 2. For each, initialize with a grid.
    # 3. Use local optimization to maximize sum of radii while maintaining non-overlap and bounds.
    
    aspect_ratios = [1.0, 1.5, 0.5, 2.0, 0.33, 3.0, 1.2, 0.8]
    
    for r in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * r / (1.0 + r)
        h = 2.0 / (1.0 + r)
        
        # Initialize grid
        # Find best grid layout for N in w x h
        # Try different grid dimensions
        best_init = None
        best_init_val = 0.0
        
        for cols in range(1, 10):
            rows = int(np.ceil(N / cols))
            # Adjust to fit N
            # This is a rough init
            
            # Calculate radii
            # dx = w / cols, dy = h / rows
            # r = min(dx, dy) / 2 - small margin
            # But we need to place exactly N circles.
            # If cols * rows > N, we just take the first N.
            
            xs = []
            ys = []
            for i in range(rows):
                for j in range(cols):
                    if len(xs) >= N:
                        break
                    x = (j + 0.5) * (w / cols)
                    y = (i + 0.5) * (h / rows)
                    xs.append(x)
                    ys.append(y)
            
            if len(xs) < N:
                continue
                
            # Initial radius estimate
            # Max possible radius is limited by min spacing
            # Simple estimate:
            r_est = min(w/cols, h/rows) / 2.0 * 0.95
            
            circles = np.zeros((N, 3))
            circles[:, 0] = xs
            circles[:, 1] = ys
            circles[:, 2] = r_est
            
            val = evaluate(circles, w, h)
            if val > best_init_val:
                best_init_val = val
                best_init = circles.copy()
        
        if best_init is None:
            continue
            
        # Now optimize
        # Variables: x_i, y_i, r_i for i in 0..N-1
        # Objective: -sum(r_i)
        # Constraints:
        #   x_i - r_i >= 0
        #   x_i + r_i <= w
        #   y_i - r_i >= 0
        #   y_i + r_i <= h
        #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        # We can use SLSQP or trust-constr. SLSQP is robust for small N.
        # To speed up, we can fix the aspect ratio and optimize positions/radii.
        
        x0 = best_init.flatten()
        
        def objective(v):
            if time.time() > deadline:
                return 1e10
            c = v.reshape(N, 3)
            return -np.sum(c[:, 2])
        
        constraints = []
        
        # Bounds constraints
        # x_i - r_i >= 0  => -x_i + r_i <= 0
        # x_i + r_i <= w  => x_i + r_i - w <= 0
        # y_i - r_i >= 0  => -y_i + r_i <= 0
        # y_i + r_i <= h  => y_i + r_i - h <= 0
        
        # Instead of adding 4*N constraints, we can use bounds on x,y,r and check in objective?
        # No, SLSQP needs explicit constraints or bounds.
        # Bounds on r: r >= 0
        # Bounds on x: 0 <= x <= w
        # Bounds on y: 0 <= y <= h
        # But x - r >= 0 is not a simple bound.
        
        # Let's use non-linear constraints for bounds and overlaps.
        
        # To reduce overhead, we might just do a few iterations.
        
        # Define constraints for overlaps
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        # This is expensive to compute in Python loops for SLSQP.
        # Alternative: use a penalty method or just a simple gradient ascent with projection?
        # Or use `scipy.optimize.minimize` with method='SLSQP' and sparse Jacobians?
        
        # Given the small N (21), SLSQP might be okay if we vectorize the constraint evaluation.
        
        def make_constraint(i, j):
            def func(v):
                c = v.reshape(N, 3)
                xi, yi, ri = c[i]
                xj, yj, rj = c[j]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            return func
        
        # This is too slow.
        
        # Let's try a different approach:
        # Fixed aspect ratio.
        # Maximize sum of r_i.
        # We can use a simple iterative solver:
        # 1. Start with grid.
        # 2. Increase all radii by small delta.
        # 3. Resolve overlaps by pushing circles apart and reducing radii if they hit walls.
        # This is a "jiggle" algorithm.
        
        # Or, use `scipy.optimize.minimize` with a single constraint function that returns an array of violations?
        # SLSQP can handle vector-valued constraints if specified as 'ineq' with array return?
        # Actually, SLSQP expects scalar constraints or a list of them.
        
        # Let's stick to a simpler heuristic if optimization is too slow.
        # Or, limit the number of optimization steps.
        
        # Let's try a simple local search:
        # Step 1: Place circles in grid.
        # Step 2: Calculate max possible radius for each circle based on neighbors and walls.
        # Step 3: Set radii to this max.
        # Step 4: Repeat with different initial aspect ratios.
        
        # This "max radius" calculation is:
        # For each circle i:
        #   r_i_max = min(
        #       x_i, w - x_i, y_i, h - y_i,  # walls
        #       min_{j != i} (dist(i, j) - r_j) / 2  # neighbors (assuming symmetric growth)
        #   )
        # But this assumes we can grow all circles simultaneously.
        # A better way:
        #   r_i = min(
        #       x_i, w - x_i, y_i, h - y_i,
        #       min_{j != i} (dist(i, j) - r_j)  # if we fix other radii?
        #   )
        
        # Let's do a few iterations of:
        #   For i in range(N):
        #       r_i = min(
        #           x_i, w - x_i, y_i, h - y_i,
        #           min_{j != i} (dist(i, j) - r_j)
        #       )
        #       r_i = max(r_i, 0)
        #   This is a fixed-point iteration.
        
        circles = best_init.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            new_radii = circles[:, 2].copy()
            for i in range(N):
                xi, yi, ri = circles[i]
                # Walls
                max_r = min(xi, w - xi, yi, h - yi)
                # Neighbors
                for j in range(N):
                    if i == j: continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    # We need dist >= ri + rj => ri <= dist - rj
                    max_r = min(max_r, dist - rj)
                new_radii[i] = max(0, max_r)
            circles[:, 2] = new_radii
            
        val = evaluate(circles, w, h)
        if val > best_val:
            best_val = val
            best_circles = circles.copy()
    
    if best_circles is None:
        # Fallback: simple grid in square
        side = 2.0
        cols = 5
        rows = 5
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N: break
                x = (j + 0.5) * (side / cols)
                y = (i + 0.5) * (side / rows)
                r = min(side/cols, side/rows) / 2.0 * 0.9
                circles.append([x, y, r])
        return np.array(circles[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
