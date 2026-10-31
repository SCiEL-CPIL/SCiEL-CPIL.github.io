"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
import time

N = 21

# EVOLVE-BLOCK-START

def compute_radii(positions, w, h):
    """
    Compute the maximum possible radii for each circle given positions,
    such that circles do not overlap and stay within the rectangle.
    Returns an array of radii.
    """
    n = len(positions)
    radii = np.zeros(n)
    
    # Distance to boundaries
    dists_bound = np.minimum(positions, np.array([w, h]) - positions)
    dists_bound = np.min(dists_bound, axis=1)
    
    # Distance to other circles
    # To be efficient, we can just compute pairwise distances
    # Since N=21, O(N^2) is fine.
    for i in range(n):
        min_dist = dists_bound[i]
        for j in range(n):
            if i == j: continue
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
        radii[i] = min_dist / 2.0
        
    return radii

def objective(params, w, h):
    """
    Objective function for optimization: negative sum of radii.
    params: flattened array of positions [x0, y0, x1, y1, ...]
    """
    positions = params.reshape(-1, 2)
    
    # Penalize if circles go out of bounds (though we can clip or constrain)
    # It's better to keep them inside. We can add a penalty if they are close to boundary
    # but the radius calculation naturally limits them.
    # However, if a circle is at x=0, radius is 0.
    
    radii = compute_radii(positions, w, h)
    
    # Add a small penalty for very small radii to encourage spreading?
    # No, just maximize sum.
    
    return -np.sum(radii)

def construct():
    start_time = time.time()
    best_result = None
    best_sum = -1
    
    # Try different aspect ratios
    # w + h = 2
    # w from 0.5 to 1.5 (symmetric, so 0.5 to 1.0 is enough)
    aspect_ratios = np.linspace(0.5, 1.0, 10)
    
    for w_frac in aspect_ratios:
        if time.time() - start_time > 100: # Leave 20s for final processing
            break
            
        w = 2 * w_frac
        h = 2 - w
        
        # Initial guess: Grid
        # Determine grid dimensions
        # Approximate number of circles per row/col
        # Let's try a few grid shapes
        best_local_sum = -1
        best_local_pos = None
        
        # Grid shapes that are close to sqrt(N)
        # N=21
        # 4x6, 5x5, 6x4, 3x7, 7x3, 2x11, 11x2
        grid_shapes = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3)]
        
        for nx, ny in grid_shapes:
            if nx * ny < N:
                continue
            
            # Create grid points
            # We want to place N points.
            # We can use the first N points of a nx x ny grid.
            # Or better, use Hungarian algorithm to pick the best N points from a larger grid?
            # Let's just use a regular grid for initialization.
            
            xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
            ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
            
            # Generate all grid points
            all_points = []
            for i in range(nx):
                for j in range(ny):
                    all_points.append([xs[i], ys[j]])
            
            all_points = np.array(all_points)
            
            # We need to pick N points.
            # To maximize the sum of radii, we want points to be as far apart as possible.
            # This is the Maximum Scattering problem.
            # We can approximate this by using the Hungarian algorithm on a cost matrix
            # where cost[i, j] is -distance to nearest other point? No, that's complex.
            # Simpler: Just take the first N points, or use a random subset?
            # Let's try to use Hungarian algorithm to assign N circles to N best grid cells.
            # But which grid cells?
            
            # Alternative: Use the grid points directly if count matches, otherwise sub-sample.
            # For now, let's just use the first N points from the grid.
            # This might be suboptimal if the grid is larger.
            
            # Better initialization: Random points within the rectangle, then optimize.
            # Or use the grid points.
            
            # Let's use the grid points as initial positions.
            # If nx*ny > N, we can randomly select N points from the grid.
            # But random selection might be bad.
            # Let's try to select points that maximize minimum distance.
            # Greedy: Pick the point with max min-distance to already picked points.
            
            selected_indices = []
            remaining = list(range(len(all_points)))
            
            # Start with a corner or center? Center is usually better for packing.
            # Let's start with the center-most point.
            center = np.array([w/2, h/2])
            dists_to_center = np.linalg.norm(all_points - center, axis=1)
            start_idx = np.argmin(dists_to_center)
            selected_indices.append(start_idx)
            remaining.remove(start_idx)
            
            while len(selected_indices) < N:
                # For each remaining point, calculate min distance to selected points
                best_score = -1
                best_idx = -1
                for idx in remaining:
                    p = all_points[idx]
                    min_d = np.min(np.linalg.norm(all_points[selected_indices] - p, axis=1))
                    # We want to maximize this min distance
                    if min_d > best_score:
                        best_score = min_d
                        best_idx = idx
                
                if best_idx == -1: break
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
                
                if time.time() - start_time > 105:
                    break

            if len(selected_indices) < N:
                continue

            initial_positions = all_points[selected_indices]
            
            # Optimize positions using Nelder-Mead
            # Flatten positions
            x0 = initial_positions.flatten()
            
            try:
                # Nelder-Mead is derivative-free and handles non-smooth functions (like min distance) well
                res = minimize(objective, x0, args=(w, h), method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.success or res.fun < best_local_sum:
                    opt_positions = res.x.reshape(-1, 2)
                    radii = compute_radii(opt_positions, w, h)
                    current_sum = np.sum(radii)
                    
                    if current_sum > best_local_sum:
                        best_local_sum = current_sum
                        best_local_pos = opt_positions
                        best_local_radii = radii
                        
            except Exception as e:
                # If optimization fails, use initial grid
                radii = compute_radii(initial_positions, w, h)
                current_sum = np.sum(radii)
                if current_sum > best_local_sum:
                    best_local_sum = current_sum
                    best_local_pos = initial_positions
                    best_local_radii = radii

        # Update global best
        if best_local_pos is not None and best_local_sum > best_sum:
            best_sum = best_local_sum
            result = np.zeros((N, 3))
            result[:, 0] = best_local_pos[:, 0]
            result[:, 1] = best_local_pos[:, 1]
            result[:, 2] = best_local_radii
            best_result = result
            
    if best_result is None:
        # Fallback
        w = 1.0
        h = 1.0
        positions = np.random.rand(N, 2) * np.array([w, h])
        radii = compute_radii(positions, w, h)
        result = np.zeros((N, 3))
        result[:, 0] = positions[:, 0]
        result[:, 1] = positions[:, 1]
        result[:, 2] = radii
        best_result = result
        
    return best_result

# EVOLVE-BLOCK-END
