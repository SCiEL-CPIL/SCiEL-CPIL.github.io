"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    best_score = -np.inf
    best_centers = None
    
    # Multi-start: Try a few different initial configurations
    configs = []
    
    # Config 1: Pure Grid 5x5x2
    nx, ny, nz = 5, 5, 2
    c1 = np.zeros((N, 3))
    idx = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                c1[idx, 0] = -0.5 + (x + 0.5) * (1.0 / nx)
                c1[idx, 1] = -0.5 + (y + 0.5) * (1.0 / ny)
                c1[idx, 2] = -0.5 + (z + 0.5) * (1.0 / nz)
                idx += 1
    configs.append(c1)
    
    # Config 2: Grid with noise
    rng = np.random.default_rng(42)
    c2 = c1 + rng.normal(0, 0.01, (N, 3))
    c2 = np.clip(c2, -0.45, 0.45)
    configs.append(c2)
    
    # Config 3: Random initialization
    c3 = rng.uniform(-0.45, 0.45, (N, 3))
    configs.append(c3)
    
    def optimize(centers):
        iterations = 1000
        for it in range(iterations):
            # Decay step size
            step = 0.01 * (0.99 ** it)
            
            # Calculate pairwise differences
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
            
            # Force magnitude: 1/d^2
            inv_dist_sq = 1.0 / (dist ** 2)
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # Sum of forces from other spheres
            forces = np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
            
            # Wall forces
            d_neg = centers + 0.5
            d_pos = 0.5 - centers
            
            d_neg_safe = np.maximum(d_neg, 1e-6)
            d_pos_safe = np.maximum(d_pos, 1e-6)
            
            f_neg = 1.0 / (d_neg_safe ** 2)
            f_pos = 1.0 / (d_pos_safe ** 2)
            
            wall_forces = f_neg - f_pos
            total_forces = forces + wall_forces
            
            # Update
            centers += step * total_forces
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    def score(centers):
        # Calculate min pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Calculate min wall distance
        min_wall = np.inf
        for k in range(3):
            min_wall = min(min_wall, np.min(centers[:, k] + 0.5), np.min(0.5 - centers[:, k]))
            
        return min(min_pair / 2.0, min_wall)
        
    for c in configs:
        optimized = optimize(c.copy())
        s = score(optimized)
        if s > best_score:
            best_score = s
            best_centers = optimized
            
    return np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
# EVOLVE-BLOCK-END
