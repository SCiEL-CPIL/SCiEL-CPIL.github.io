import numpy as np
import time

def construct():
    N = 100
    
    # Initial configuration: 5x5x4 grid
    # This is a good starting point for 100 spheres
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small noise to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    
    start_time = time.time()
    max_time = 115.0 # Leave 5s margin
    
    step_size = 0.002
    decay = 0.999
    max_iter = 30000
    
    best_radius = -np.inf
    best_centers = centers.copy()
    
    for i in range(max_iter):
        if time.time() - start_time > max_time:
            break
            
        # 1. Pairwise distances and forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists[range(N), range(N)] = np.inf
        
        # Force: sum (x_i - x_j) / dist^2
        inv_dist2 = 1.0 / (dists**2 + 1e-12)
        inv_dist2[range(N), range(N)] = 0.0
        
        # Limit force magnitude to prevent instability
        max_force = 50.0
        inv_dist2 = np.minimum(inv_dist2, max_force)
        
        forces = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
        
        # 2. Wall forces
        # Current estimated radius
        min_dist = np.min(dists)
        r_est = min_dist / 2.0
        
        # Limit for centers: 0.5 - r_est
        limit = 0.5 - r_est
        
        wall_forces = np.zeros_like(centers)
        for k in range(3):
            coords = centers[:, k]
            abs_coords = np.abs(coords)
            # Violation of the soft boundary
            viol = np.maximum(0, abs_coords - limit)
            # Force direction: towards 0
            direction = -np.sign(coords)
            # Force magnitude: proportional to violation, with high stiffness
            wall_forces[:, k] = direction * viol * 100.0
            
        total_forces = forces + wall_forces
        
        # 3. Update
        centers += step_size * total_forces
        
        # 4. Hard clamp to stay within cube [-0.5, 0.5]
        centers = np.clip(centers, -0.5, 0.5)
        
        # 5. Evaluate every 5 iterations
        if i % 5 == 0:
            diff_eval = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_eval = np.linalg.norm(diff_eval, axis=2)
            dists_eval[range(N), range(N)] = np.inf
            min_dist_eval = np.min(dists_eval)
            
            wall_min = np.min(0.5 - np.abs(centers))
            
            r = min(min_dist_eval / 2.0, wall_min)
            
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
        # Decay step size
        step_size *= decay
        
    return best_centers
