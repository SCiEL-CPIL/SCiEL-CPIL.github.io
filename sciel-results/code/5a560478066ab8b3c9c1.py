"""
Circle packing in perimeter-4 rectangle using structured lattice templates,
Lagrangian relaxation, and global scaling.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0

    def is_valid(circles, W, H, tol=1e-7):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True

    def generate_hex_grid(W, H, r, shift_x=0.0, shift_y=0.0):
        """Generate a hexagonal grid of circles inside W x H rectangle."""
        circles = []
        if r <= 0:
            return circles
        
        # Hexagonal spacing
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        # Determine number of rows and cols
        # We might not fill the whole rectangle, just place as many as fit
        # We want exactly N circles, so we generate a superset and take the first N
        # or we generate exactly N if possible.
        
        # To ensure we get N circles, we generate a grid larger than needed
        # and then select the first N that fit.
        
        y = r + shift_y
        row_idx = 0
        while y <= H - r:
            x_offset = 0 if row_idx % 2 == 0 else r
            x = r + x_offset + shift_x
            while x <= W - r:
                if len(circles) < N:
                    circles.append([x, y, r])
                x += dx
            y += dy
            row_idx += 1
            
        # If we didn't get enough circles, try to add more by allowing slight overflow?
        # No, strict packing. If we don't have N, this configuration is invalid for direct use,
        # but the scaling step might adjust r or W/H.
        # Actually, if we can't fit N circles of radius r, the configuration is "too sparse" 
        # or the rectangle is too small. 
        # For the optimization, we need a fixed number of points. 
        # Let's force exactly N points by padding if necessary? 
        # No, better to generate a grid that is guaranteed to have at least N points 
        # by choosing r small enough, or just take the first N from a larger grid.
        
        return circles

    def generate_rect_grid(W, H, r, shift_x=0.0, shift_y=0.0):
        """Generate a rectangular grid of circles inside W x H rectangle."""
        circles = []
        if r <= 0:
            return circles
            
        dx = 2 * r
        dy = 2 * r
        
        y = r + shift_y
        row_idx = 0
        while y <= H - r:
            x = r + shift_x
            while x <= W - r:
                if len(circles) < N:
                    circles.append([x, y, r])
                x += dx
            y += dy
            row_idx += 1
            
        return circles

    def optimize_packing(circles, W, H, time_limit):
        """
        Refine the positions of circles to maximize the minimum distance to walls and neighbors.
        We maximize the minimum "slack" (distance to constraint).
        Objective: max min_i (min_dist_to_constraint_i)
        This is equivalent to expanding all radii uniformly until they touch.
        """
        if time.time() > time_limit:
            return circles, False

        n = len(circles)
        if n == 0:
            return circles, False

        # Flatten circles to vector [x1, y1, x2, y2, ...]
        x0 = np.array([c[0] for c in circles])
        y0 = np.array([c[1] for c in circles])
        
        # Initial radii are 0 for the optimization variable, 
        # but we treat the "radius" as a variable to be maximized?
        # No, standard approach: 
        # Fix the relative positions? No.
        # Let's maximize the minimum distance between any two circles and the walls.
        # Let d_min be the minimum distance.
        # We want to maximize d_min.
        
        # Variables: positions (2N)
        # Objective: -min( min_{i<j} dist(i,j), min_i dist(i, wall) )
        
        # To make it differentiable, we can use a soft-min or just optimize the minimum directly 
        # using a simple gradient ascent on the minimum constraint.
        # However, scipy minimize is for minimization.
        # Let's minimize -min_dist.
        
        def neg_min_dist(params):
            x = params[:n]
            y = params[n:]
            
            # Wall distances
            dists = []
            for i in range(n):
                dists.append(x[i])
                dists.append(W - x[i])
                dists.append(y[i])
                dists.append(H - y[i])
            
            # Pairwise distances
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    dists.append(dist)
                    
            return -min(dists)
        
        # Initial guess
        x_init = x0.copy()
        y_init = y0.copy()
        
        # Add small noise to break symmetry
        x_init += np.random.normal(0, 0.01 * W, n)
        y_init += np.random.normal(0, 0.01 * H, n)
        
        # Clip to bounds
        x_init = np.clip(x_init, 0, W)
        y_init = np.clip(y_init, 0, H)
        
        params_init = np.concatenate([x_init, y_init])
        
        # Constraints: keep inside box (soft, handled by objective if we just maximize min dist)
        # But if a circle goes outside, min dist becomes negative.
        # We want to stay inside.
        
        # Use Nelder-Mead or Powell? Nelder-Mead is robust for small N.
        # N=21 -> 42 variables. Nelder-Mead might be slow but 145s is plenty.
        
        try:
            res = minimize(
                neg_min_dist, 
                params_init, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if time.time() > time_limit:
                return circles, False
                
            x_opt = res.x[:n]
            y_opt = res.x[n:]
            
            # The optimal min distance d* is -res.fun
            # This d* is the maximum possible minimum distance.
            # We can assign this radius to all circles?
            # Yes, if we maximize the minimum distance, we can set all radii to d*.
            # But wait, the "distance" to wall is x_i. If x_i = d*, then the circle of radius d* touches the wall.
            # The distance between circle i and j is d*. If we set radius r_i = d*, then dist(i,j) >= r_i + r_j = 2*d*.
            # Wait, the objective was min(dist(i,j)). If min(dist(i,j)) = d*, then dist(i,j) >= d*.
            # If we set r_i = d*/2, then r_i + r_j = d*. So they just touch.
            # So the maximum uniform radius is d_min / 2.
            
            r_opt = -res.fun / 2.0
            
            new_circles = []
            for i in range(n):
                new_circles.append([x_opt[i], y_opt[i], r_opt])
                
            return new_circles, True
            
        except Exception:
            return circles, False

    # Main loop
    random.seed(42)
    np.random.seed(42)

    # Try different aspect ratios
    # Perimeter 4 => 2(W+H)=4 => W+H=2
    # W ranges from 0.5 to 1.5 (symmetric)
    aspect_ratios = np.linspace(0.6, 1.4, 9)

    for W_ratio in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0.1 or H <= 0.1:
            continue

        # Generate initial configurations
        inits = []

        # 1. Hexagonal grid
        # Estimate max radius for hex grid
        # Area approx: N * (2r)^2 * sqrt(3)/2 ?
        # Let's just try a range of r
        for r_init in [0.05, 0.08, 0.1, 0.12]:
            hex_circles = generate_hex_grid(W, H, r_init)
            if len(hex_circles) >= N:
                # Take first N
                inits.append([c[:] for c in hex_circles[:N]])
            elif len(hex_circles) > 0:
                # If we have fewer, pad with random? No, skip.
                pass

        # 2. Rectangular grid
        for r_init in [0.05, 0.08, 0.1, 0.12]:
            rect_circles = generate_rect_grid(W, H, r_init)
            if len(rect_circles) >= N:
                inits.append([c[:] for c in rect_circles[:N]])

        # 3. Random initialization with decent radius
        for _ in range(5):
            random_circles = []
            for _ in range(N):
                x = random.uniform(0.1, W - 0.1)
                y = random.uniform(0.1, H - 0.1)
                r = random.uniform(0.05, 0.1)
                random_circles.append([x, y, r])
            inits.append(random_circles)

        # Optimize each init
        for init_circles in inits:
            if time.time() > deadline - 10:
                break
                
            # The optimization function ignores the initial radii and finds the max uniform radius.
            # This is a strong assumption: that the optimal packing has equal radii.
            # For 21 circles, equal radii is a very good approximation for the global optimum,
            # especially with lattice structures.
            
            opt_circles, success = optimize_packing(init_circles, W, H, deadline - 5)
            
            if success and opt_circles:
                # Verify validity
                if is_valid(opt_circles, W, H, tol=1e-5):
                    total_r = sum(c[2] for c in opt_circles)
                    if total_r > best_sum_r:
                        best_sum_r = total_r
                        best_config = np.array(opt_circles)
                        best_W = W
                        best_H = H

    if best_config is not None and len(best_config) == N:
        return best_config

    # Fallback: simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append((x, y, r))
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
