import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # 1. Initialization
    # Start with a 7x7 grid (49 points)
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We need 50 points. Add a 50th point.
    # The "holes" in a square grid are at (i+0.5, j+0.5) relative to grid nodes?
    # Actually, just place the 50th point at the center of the largest empty space.
    # For a 7x7 grid, the center is (0,0) if m is odd?
    # ticks for m=7: 
    # 0.5/7 - 0.5 = -0.428
    # 1.5/7 - 0.5 = -0.285
    # ...
    # 3.5/7 - 0.5 = 0.0
    # So (0,0) is a grid point.
    # The best spot for the 50th point in a 7x7 grid is likely one of the 4 central holes?
    # Or we can just initialize 50 points in a slightly perturbed hexagonal pattern.
    
    # Let's try a hexagonal lattice initialization for 50 points.
    # Hex packing is denser.
    # Rows: 
    # Row 0: 7 points
    # Row 1: 7 points (shifted)
    # ...
    # Total points approx sqrt(N).
    
    # Simpler: Use the 7x7 grid (49 points) and add the 50th at a random valid position, 
    # then let the optimizer fix it.
    # Better: Add the 50th at the center of the square, but shift the grid slightly?
    # No, just add it at (0, 0) and let the optimizer push the center point out?
    # The center of the 7x7 grid is occupied.
    # Let's pick the point (0.0, 0.0) as the 50th, but we already have a point there?
    # ticks: 3.5/7 - 0.5 = 0.0. Yes, (0,0) is in the grid.
    # So we cannot just add (0,0).
    
    # Let's generate 50 points using a "hexagonal" style grid.
    # Number of rows: 7
    # Points per row: 7, 6, 7, 6, 7, 6, 7? 
    # 7+6+7+6+7+6+7 = 46. Not enough.
    # 8, 7, 8, 7, 8, 7, 8?
    # 8+7+8+7+8+7+8 = 53. Too many.
    
    # Let's stick to the 7x7 grid + 1 point.
    # Where is the best spot for the 50th point?
    # The "holes" in a square grid are at half-integer coordinates relative to the grid.
    # Grid x: -0.428, -0.285, -0.142, 0, 0.142, 0.285, 0.428
    # Hole x: -0.357, -0.214, -0.071, 0.071, 0.214, 0.357
    # Hole y: same.
    # A hole like (0.071, 0.071) is equidistant to 4 grid points.
    # Distance to neighbors: sqrt(0.071^2 + 0.071^2) = 0.10.
    # Grid spacing is 0.142.
    # So this point is closer to its neighbors than the grid points are to each other.
    # This will reduce the min distance.
    
    # Alternative: Perturb the 7x7 grid to make room for the 50th.
    # Initialize 50 points:
    # 49 in grid, 1 at a random hole.
    centers = np.vstack([grid, np.array([[0.0714, 0.0714]])])
    
    # 2. Optimization
    # We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    # Use a simple gradient ascent on the "soft min" or iterative push.
    
    def get_score(centers):
        # Pairwise distances
        # N=50, so 1225 pairs. Fast enough.
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Wall distances
        # min(0.5 - |x|, 0.5 - |y|)
        wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
        min_wall = np.min(wall_dists)
        
        return np.minimum(min_pair / 2.0, min_wall)

    # Iterative Push-Apart
    # To avoid O(N^2) distance calculation in every iteration if N was larger, 
    # but for N=50 it's fine.
    
    # We will use a "temperature" annealing approach.
    # Move points in the direction of the gradient of the min distance.
    
    best_centers = centers.copy()
    best_score = get_score(best_centers)
    
    # Number of iterations
    # We have ~140s.
    # Each iteration: O(N^2) ~ 2500 ops. Very fast.
    # We can do many iterations.
    
    T = 0.01
    step_size = 0.005
    
    for i in range(20000): # Cap iterations
        if time.time() > deadline:
            break
        
        # Calculate distances
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find the closest pair
        min_idx = np.argmin(dists)
        i_idx, j_idx = np.unravel_index(min_idx, (N, N))
        
        # Also check walls
        wall_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = np.minimum(dists[i_idx, j_idx] / 2.0, wall_dists[min_wall_idx])
        
        if current_r > best_score:
            best_score = current_r
            best_centers = centers.copy()
            
        # Update strategy:
        # Push the closest pair apart.
        # Also push points against walls away from walls.
        
        # Vectorized update for all points?
        # Better: Calculate a "force" on each point.
        # Force from pairs:
        forces = np.zeros_like(centers)
        
        # To make it efficient, we can just update the specific pair and the wall-violating point.
        # But local minima are a risk.
        # Let's use a "soft" repulsion for all pairs within a threshold.
        
        # Threshold: 1.5 * current_r
        thresh = 1.5 * current_r + 1e-6
        mask = (dists < thresh) & (dists > 1e-6)
        
        # Normalize differences
        # dists is symmetric.
        for u in range(N):
            for v in range(N):
                if mask[u, v]:
                    d = dists[u, v]
                    if d < 1e-6:
                        # Random nudge
                        force = np.random.randn(2) * 0.01
                    else:
                        force = (centers[u] - centers[v]) / d * (1.0 / d) # Inverse square? Or just direction?
                        # Simple direction:
                        force = (centers[u] - centers[v]) / d
                    forces[u] += force
                    # forces[v] -= force # Symmetric, but we loop all, so we double count?
                    # If we loop all u,v, we add force to u from v, and v from u.
                    # This is fine.
        
        # Wall forces
        for u in range(N):
            # X wall
            if centers[u, 0] > 0.5 - current_r:
                forces[u, 0] -= 1.0
            elif centers[u, 0] < -0.5 + current_r:
                forces[u, 0] += 1.0
            
            # Y wall
            if centers[u, 1] > 0.5 - current_r:
                forces[u, 1] -= 1.0
            elif centers[u, 1] < -0.5 + current_r:
                forces[u, 1] += 1.0
                
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms[:, np.newaxis]
        
        # Update
        # Add some noise to escape local minima
        noise = np.random.randn(N, 2) * T
        centers += forces * step_size + noise
        
        # Project back to square (hard constraint)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Anneal temperature
        T *= 0.999
        if T < 0.001:
            T = 0.001

    return best_centers
