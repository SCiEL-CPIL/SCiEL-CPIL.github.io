import numpy as np

def construct():
    N = 50
    # Initialize with 7x7 grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Add 50th point
    extra = np.array([[1/14, 1/14]])
    points = np.vstack([grid, extra])
    
    # Add small random noise
    np.random.seed(42)
    points += np.random.uniform(-0.001, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    # Optimization
    for i in range(20000):
        # Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find nearest neighbor for each point
        min_pair_idx = np.argmin(dists, axis=1)
        
        # Calculate wall distances
        wall_dists_full = np.stack([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ], axis=-1)
        min_wall_idx = np.argmin(wall_dists_full, axis=1)
        
        # Calculate forces
        j_indices = min_pair_idx
        vec_pair = points - points[j_indices]
        norms = np.linalg.norm(vec_pair, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vec_pair = vec_pair / norms
        
        wall_vectors = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])
        vec_wall = wall_vectors[min_wall_idx]
        
        forces = vec_pair + vec_wall
        
        # Adaptive step size
        if i < 10000:
            step_size = 0.01
        elif i < 15000:
            step_size = 0.005
        else:
            step_size = 0.001
            
        points += step_size * forces
        points = np.clip(points, -0.5, 0.5)
        
    return points
