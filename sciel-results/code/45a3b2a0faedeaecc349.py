import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    def compute_radius(centers):
        # 1. Distance to boundary
        # For each point, distance to nearest edge is 0.5 - max(|x|, |y|)
        # We need the minimum of this over all points.
        max_abs_coords = np.max(np.abs(centers), axis=1)
        min_boundary_dist = np.min(0.5 - max_abs_coords)
        
        # 2. Distance between centers
        x = centers[:, 0]
        y = centers[:, 1]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx*dx + dy*dy)
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_center_dist = np.min(dists)
        
        # Radius is half the min center distance, or the min boundary distance
        return min(min_boundary_dist, min_center_dist / 2.0)

    # 1. Initialization: Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_x, grid_y = np.meshgrid(ticks, ticks)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    best_centers = grid[:N].copy()
    best_radius = compute_radius(best_centers)
    
    # 2. Simulated Annealing
    centers = best_centers.copy()
    current_radius = best_radius
    
    # Parameters
    current_temp = 0.1
    min_temp = 1e-6
    cooling_rate = 0.9995
    base_step = 0.2
    
    # Run until deadline
    while time.time() < deadline:
        # Pick a random circle to move
        i = np.random.randint(N)
        
        # Determine step size based on temperature
        step = base_step * (current_temp / 0.1)
        
        old_pos = centers[i].copy()
        
        # Apply random move
        centers[i, 0] += np.random.uniform(-step, step)
        centers[i, 1] += np.random.uniform(-step, step)
        
        # Enforce boundary constraints
        centers[i, 0] = np.clip(centers[i, 0], -0.5, 0.5)
        centers[i, 1] = np.clip(centers[i, 1], -0.5, 0.5)
        
        # Calculate new radius
        new_radius = compute_radius(centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            # Accept worse solution with probability exp((new_radius - current_radius) / current_temp)
            delta = new_radius - current_radius
            # delta is negative or zero
            prob = np.exp(delta / current_temp)
            if np.random.rand() < prob:
                current_radius = new_radius
            else:
                # Revert move
                centers[i] = old_pos
                # current_radius remains unchanged
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp

    # Ensure final result is valid (clipped)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
