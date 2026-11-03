"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Uses a Repulsion-based Local Relaxation algorithm.
    Unlike Simulated Annealing which minimizes a global energy, this method
    directly maximizes the minimum distance by iteratively resolving the
    tightest constraints (nearest neighbors and walls).
    """
    rng = np.random.RandomState(123)
    
    # Initial guess: Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(float)
    # Add jitter to break symmetry
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    # Ensure within bounds initially
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_r = 0.0
    
    # Helper to compute the minimum radius for a set of points
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        min_wall_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r
    
    best_r = compute_radius(points)
    
    n_iter = 50000
    base_step = 0.05
    
    for i in range(n_iter):
        # Adaptive step size
        step = base_step * (1.0 - i / n_iter) + 0.001
        
        # Calculate forces/repulsions
        # 1. Find nearest neighbor for each point
        # 2. Find nearest wall for each point
        
        # Pairwise
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Nearest neighbor index and distance for each point
        min_dist_idx = np.argmin(dists, axis=1)
        min_dist_val = np.min(dists, axis=1)
        
        # Wall constraints
        # Distance to each wall
        d_left = points[:, 0] + 0.5
        d_right = 0.5 - points[:, 0]
        d_bottom = points[:, 1] + 0.5
        d_top = 0.5 - points[:, 1]
        
        # Force vector for each point
        forces = np.zeros_like(points)
        
        for j in range(N):
            # Nearest neighbor force
            nn_idx = min_dist_idx[j]
            if min_dist_val[j] < 0.15: # Only repel if close enough
                # Vector from j to nn
                vec = points[nn_idx] - points[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    forces[j] += (vec / norm) * (0.15 - min_dist_val[j])
            
            # Wall forces
            # If close to left wall
            if d_left[j] < 0.05:
                forces[j, 0] += (0.05 - d_left[j])
            # If close to right wall
            if d_right[j] < 0.05:
                forces[j, 0] -= (0.05 - d_right[j])
            # If close to bottom wall
            if d_bottom[j] < 0.05:
                forces[j, 1] += (0.05 - d_bottom[j])
            # If close to top wall
            if d_top[j] < 0.05:
                forces[j, 1] -= (0.05 - d_top[j])
        
        # Update points
        new_points = points + forces * step
        
        # Add random noise to escape local minima
        # Noise magnitude decays
        noise_scale = 0.02 * (1.0 - i / n_iter) + 0.001
        new_points += rng.normal(0, noise_scale, size=new_points.shape)
        
        # Clip to bounds
        new_points = np.clip(new_points, -0.5, 0.5)
        
        # Check validity and score
        # Note: We must ensure no overlap in the final result, but during optimization
        # we allow temporary overlaps if the force pushes them apart.
        # However, the score is based on the final configuration.
        
        r_new = compute_radius(new_points)
        
        if r_new > best_r:
            best_r = r_new
            best_points = new_points.copy()
        
        points = new_points
    
    return best_points
# EVOLVE-BLOCK-END
