import time
import numpy as np
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Objective: Minimize sum of squared overlaps for a given radius r
    # Overlap for pair i,j: max(0, r - dist(i,j)/2)^2
    # Overlap for boundary i: max(0, r - dist_to_boundary(i))^2
    # dist_to_boundary is min(0.5 - x_i, 0.5 - y_i, x_i + 0.5, y_i + 0.5)
    
    def overlap_energy(positions, r):
        # positions: (N, 2)
        # Compute pairwise distances
        # For N=200, 200*200/2 = 20000 pairs. This is fast.
        # Using cdist is efficient.
        from scipy.spatial.distance import cdist
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, 2*r + 1) # Ensure diagonal doesn't contribute
        
        # Pairwise overlaps
        # We want max(0, r - d/2)^2
        # d/2 is the distance between centers. If d < 2r, they overlap.
        # Overlap amount = 2r - d.
        # Squared overlap = (2r - d)^2 if d < 2r else 0.
        
        # Vectorized calculation
        # overlap = np.maximum(0, 2*r - dists)
        # energy = np.sum(overlap**2)
        
        # Boundary overlaps
        # Distance to boundary for each point
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # dist_x = min(0.5 - x, x + 0.5)
        # dist_y = min(0.5 - y, y + 0.5)
        # dist_b = min(dist_x, dist_y)
        # Overlap = max(0, r - dist_b)^2
        
        xs = positions[:, 0]
        ys = positions[:, 1]
        
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        dist_b = np.minimum(dist_x, dist_y)
        
        boundary_overlap = np.maximum(0, r - dist_b)
        boundary_energy = np.sum(boundary_overlap**2)
        
        # Pairwise
        # Only consider upper triangle to avoid double counting? 
        # cdist gives full matrix. Sum of squares will double count. 
        # Since we just want to minimize, factor 2 doesn't matter for optimization direction.
        pair_overlap = np.maximum(0, 2*r - dists)
        pair_energy = np.sum(pair_overlap**2)
        
        return pair_energy + boundary_energy

    def hexagonal_init(n):
        # Create a hexagonal grid that fits in the square
        # Approximate density: 200 circles in 1x1 square.
        # Area per circle ~ 1/200 = 0.005.
        # Hexagonal packing area per circle = (sqrt(3)/2) * d^2 where d is center distance.
        # d ~ sqrt(2*0.005/sqrt(3)) ~ 0.075? 
        # Radius r ~ d/2 ~ 0.037. Matches known best.
        
        # Let's just generate a grid of hex points and center it.
        # Spacing s.
        # Rows: y = i * s * sqrt(3)/2
        # Cols: x = j * s + (i%2)*s/2
        
        # Estimate s from known radius ~ 0.0366 -> d = 2r = 0.0732
        s = 0.074 
        points = []
        # Generate enough points
        for i in range(-10, 10):
            for j in range(-10, 10):
                x = j * s + (i % 2) * s / 2
                y = i * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        # Filter points roughly in the square
        mask = (points[:, 0] > -0.5) & (points[:, 0] < 0.5) & (points[:, 1] > -0.5) & (points[:, 1] < 0.5)
        points = points[mask]
        
        # If we have too many, take the first N (or random N)
        if len(points) > n:
            # Shuffle and take N
            idx = np.random.choice(len(points), n, replace=False)
            points = points[idx]
        elif len(points) < n:
            # If too few, add random points in the square
            needed = n - len(points)
            extra = np.random.uniform(-0.4, 0.4, size=(needed, 2))
            points = np.vstack([points, extra])
            
        return points

    def square_grid_init(n):
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]

    best_radius = -np.inf
    best_positions = None
    
    # Initializations
    inits = [
        hexagonal_init(N),
        square_grid_init(N),
        np.random.uniform(-0.4, 0.4, size=(N, 2))
    ]
    
    # Inflation method
    # Start with a small radius and increase it.
    # We need to find the maximum r for which a feasible configuration exists.
    
    # We will try to inflate from a base radius.
    # The known best is ~0.0366.
    # Let's start at 0.03 and go up.
    
    for init_pos in inits:
        if time.time() > deadline:
            break
            
        # Start with a radius slightly smaller than expected
        r = 0.03
        # Step size for inflation
        step = 0.001
        
        # We will run a local search for each r.
        # To save time, we can reuse the previous positions as the start for the next r.
        current_pos = init_pos.copy()
        
        # First, ensure current_pos is feasible for r=0.03. 
        # If not, we might need to shrink or just start optimizing from a smaller r.
        # For robustness, let's start at r=0.02.
        
        r = 0.02
        success = False
        
        while r < 0.045: # Upper bound guess
            if time.time() > deadline - 5:
                break
                
            # Optimize positions to minimize overlap for current r
            # Use L-BFGS-B
            try:
                result = minimize(
                    lambda x: overlap_energy(x, r),
                    current_pos,
                    method='L-BFGS-B',
                    bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )
            except Exception:
                break
                
            current_pos = result.x
            energy = result.fun
            
            # Check if feasible (energy close to 0)
            # Due to numerical issues, we allow a small tolerance
            if energy < 1e-4:
                # Feasible! Try to increase r
                r += step
                # We can reduce step size as we get closer to the limit? 
                # Or keep constant for simplicity.
            else:
                # Infeasible. We have exceeded the max radius for this configuration.
                # The best radius for this path is the previous r.
                # Let's do a binary search between r-step and r to find the exact max.
                r_low = r - step
                r_high = r
                
                for _ in range(10): # 10 iterations of binary search
                    if time.time() > deadline - 5:
                        break
                    r_mid = (r_low + r_high) / 2
                    try:
                        res_mid = minimize(
                            lambda x: overlap_energy(x, r_mid),
                            current_pos,
                            method='L-BFGS-B',
                            bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                            options={'maxiter': 50, 'ftol': 1e-6}
                        )
                        energy_mid = res_mid.fun
                        if energy_mid < 1e-4:
                            r_low = r_mid
                            current_pos = res_mid.x
                        else:
                            r_high = r_mid
                    except Exception:
                        break
                
                # r_low is the best feasible radius found
                if r_low > best_radius:
                    best_radius = r_low
                    best_positions = current_pos.copy()
                
                break # Move to next initialization
        
        # If we reached the upper bound 0.045 and were still feasible, 
        # we might have found a better packing. 
        # But 0.045 is likely too high. 
        # If the loop finished without breaking (i.e., r >= 0.045), we should check the final r.
        if r >= 0.045 and best_radius < r:
             # Verify feasibility at r
             try:
                 res_final = minimize(
                     lambda x: overlap_energy(x, r),
                     current_pos,
                     method='L-BFGS-B',
                     bounds=[(-0.5, -0.5), (0.5, 0.5)] * N,
                     options={'maxiter': 50, 'ftol': 1e-6}
                 )
                 if res_final.fun < 1e-4:
                     best_radius = r
                     best_positions = res_final.x
             except Exception:
                 pass

    # Fallback if no valid position found
    if best_positions is None:
        best_positions = square_grid_init(N)
        # Calculate radius for this fallback
        # Just return it, score will be low but valid.
        
    # Final validation and clamping
    best_positions = np.clip(best_positions, -0.4999, 0.4999)
    
    return best_positions
