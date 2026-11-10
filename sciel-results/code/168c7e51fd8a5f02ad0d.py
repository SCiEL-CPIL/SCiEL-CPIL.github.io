# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    # 1. Generate a triangular lattice
    a = 1.0
    h = a * np.sqrt(3) / 2
    
    # Generate a sufficiently large grid
    grid_size = 10
    points_grid = []
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            x = i * a + (j % 2) * a / 2
            y = j * h
            points_grid.append([x, y])
            
    points_grid = np.array(points_grid)
    
    # Center the grid
    centroid = np.mean(points_grid, axis=0)
    points_grid -= centroid
    
    # Sort by distance to origin and pick the closest 16
    dists = np.linalg.norm(points_grid, axis=1)
    indices = np.argsort(dists)
    points = points_grid[indices[:n]].copy()
    
    # 2. Optimize using repulsion
    num_iters = 5000
    alpha = 0.5
    
    for k in range(num_iters):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.linalg.norm(diff, axis=2)
        dists_matrix += 1e-9 # Prevent division by zero
        
        force_vectors = diff / (dists_matrix[:, :, np.newaxis] ** 3)
        forces = np.sum(force_vectors, axis=1)
        
        points += alpha * forces
        
        # Normalize
        center = np.mean(points, axis=0)
        points -= center
        
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 0:
            points /= d_max
            
        alpha *= 0.9995

    return points

# EVOLVE-BLOCK-END
