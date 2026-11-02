import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_centers = None
    best_radius = -1.0
    
    # Define the objective function to minimize
    # We want to maximize the minimum distance between spheres and walls.
    # Minimizing sum of inverse squared distances approximates this.
    def objective(params):
        centers = params.reshape(N, 3)
        
        # Calculate pairwise distances
        # d[i, j] is distance between center i and center j
        # We need sum_{i<j} 1/d_ij^2
        # Vectorized calculation
        # centers shape (N, 3)
        # diff shape (N, N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero for i==j
        np.fill_diagonal(dist_sq, np.inf)
        
        # Sum of 1/dist_sq for i < j
        # dist_sq is symmetric. We can sum all and divide by 2, or use triu_indices
        # Using triu_indices is safer for memory if N is large, but N=50 is small.
        iu = np.triu_indices(N, k=1)
        pair_dist_sq = dist_sq[iu]
        
        # Clip to avoid huge values if points coincide
        pair_dist_sq = np.clip(pair_dist_sq, 1e-10, None)
        
        obj_pairs = np.sum(1.0 / pair_dist_sq)
        
        # Calculate distances to walls
        # Cube is [-0.5, 0.5]^3
        # Distance to wall for point p is min(0.5 - p_x, 0.5 - p_y, 0.5 - p_z, p_x + 0.5, p_y + 0.5, p_z + 0.5)
        # Actually, distance to nearest face:
        # d_x = 0.5 - abs(center[:, 0])
        # d_y = 0.5 - abs(center[:, 1])
        # d_z = 0.5 - abs(center[:, 2])
        # dist_to_wall = min(d_x, d_y, d_z)
        
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        dz = 0.5 - np.abs(centers[:, 2])
        
        # Ensure points are inside for the penalty calculation, though bounds should handle it
        # If a point is outside, dx can be negative. 
        # The penalty 1/d^2 will explode if d is small or negative? 
        # If d is negative, d^2 is positive, so it still penalizes.
        # But we want to push them IN.
        # Let's use the actual distance to the boundary of the cube.
        # If center is at 0.6, dist to wall 0.5 is 0.1 (outside). 
        # The formula 0.5 - abs(x) gives -0.1. Squared is 0.01. 1/0.01 = 100.
        # This penalizes being outside.
        
        dist_wall_sq = np.minimum.reduce([dx**2, dy**2, dz**2])
        dist_wall_sq = np.clip(dist_wall_sq, 1e-10, None)
        
        obj_walls = np.sum(1.0 / dist_wall_sq)
        
        return obj_pairs + obj_walls

    def calculate_radius(centers):
        # Calculate the actual max radius for this configuration
        # 1. Min pairwise distance / 2
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # 2. Min distance to wall
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        dz = 0.5 - np.abs(centers[:, 2])
        min_wall_dist = np.min(np.minimum.reduce([dx, dy, dz]))
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # We will try a few random restarts and some structured starts
    # Structured start: Grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add jitter to grid
    np.random.seed(42)
    jittered_grid = grid[:N] + np.random.uniform(-0.05, 0.05, (N, 3))
    # Clip to bounds
    jittered_grid = np.clip(jittered_grid, -0.49, 0.49)
    
    starts = [jittered_grid.flatten()]
    
    # Add a few random starts
    for _ in range(10):
        random_start = np.random.uniform(-0.45, 0.45, (N, 3))
        starts.append(random_start.flatten())
        
    for start_idx, start_params in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        try:
            # L-BFGS-B is good for bound-constrained problems
            res = minimize(
                objective, 
                start_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 200}
            )
            
            if res.success or res.fun < 1e6: # Check if it didn't explode
                centers = res.x.reshape(N, 3)
                # Ensure centers are strictly inside to avoid numerical issues in radius calc
                centers = np.clip(centers, -0.5, 0.5)
                
                r = calculate_radius(centers)
                
                if r > best_radius:
                    best_radius = r
                    best_centers = centers
                    
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to grid if optimization failed completely
        return grid[:N]
        
    return best_centers
