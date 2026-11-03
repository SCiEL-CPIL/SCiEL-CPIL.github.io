import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Initialize with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    
    best_points = points.copy()
    best_radius = -1.0
    
    def compute_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        dist_to_boundary = np.min(np.stack([
            0.5 - pts[:, 0],
            pts[:, 0] + 0.5,
            0.5 - pts[:, 1],
            pts[:, 1] + 0.5
        ], axis=-1), axis=-1)
        min_wall_dist = np.min(dist_to_boundary)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_radius = compute_radius(points)
    
    # Parameters for SA / Repulsion hybrid
    T = 0.005
    T_min = 1e-7
    cooling_rate = 0.999
    
    current_points = points.copy()
    current_radius = best_radius
    
    # Pre-allocate arrays for speed
    while time.time() - start_time < time_limit:
        if T < T_min:
            if current_radius < best_radius:
                current_points = best_points.copy()
                current_radius = best_radius
            T = T_min * 10
            
        # Calculate nearest neighbors and forces
        # dists[i,j]
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find nearest neighbor for each point
        nn_idx = np.argmin(dists, axis=1)
        
        # Calculate force (repulsion)
        # Force is direction away from nearest neighbor
        # If a point is close to a wall, also add force away from wall
        forces = np.zeros_like(current_points)
        
        for i in range(N):
            nn = nn_idx[i]
            dir_to_nn = current_points[nn] - current_points[i]
            dist_to_nn = dists[i, nn]
            
            # Normalize direction
            if dist_to_nn > 1e-9:
                dir_to_nn = dir_to_nn / dist_to_nn
            else:
                dir_to_nn = np.random.normal(size=2)
                dir_to_nn /= np.linalg.norm(dir_to_nn)
            
            # Force magnitude
            # Stronger if closer
            f_mag = 1.0 / (dist_to_nn + 0.01)
            forces[i] -= dir_to_nn * f_mag # Away from NN
            
            # Wall constraints
            # If close to wall, push in
            margin = 0.01
            if current_points[i, 0] < -0.5 + margin:
                forces[i, 0] += 1.0
            elif current_points[i, 0] > 0.5 - margin:
                forces[i, 0] -= 1.0
                
            if current_points[i, 1] < -0.5 + margin:
                forces[i, 1] += 1.0
            elif current_points[i, 1] > 0.5 - margin:
                forces[i, 1] -= 1.0
        
        # Propose move
        # Step size depends on T
        step_size = T * 0.1
        noise = np.random.normal(0, step_size, (N, 2))
        new_points = current_points + forces * 0.01 + noise
        
        # Clamp
        new_points[:, 0] = np.clip(new_points[:, 0], -0.5, 0.5)
        new_points[:, 1] = np.clip(new_points[:, 1], -0.5, 0.5)
        
        new_radius = compute_radius(new_points)
        
        # Acceptance
        delta = new_radius - current_radius
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_points = new_points
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        
        T *= cooling_rate

    return best_points
