import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # Deterministic initialization
    # Start with a grid and perturb slightly, or use a random seed
    np.random.seed(42)
    
    # Initialize with a 5x5x5 grid subset? 125 points.
    # Or just random points in the cube.
    # Random points are often better for local search if we do many iterations.
    centers = np.random.uniform(-0.4, 0.4, (N, 3))
    
    # Objective function to minimize: negative of the minimum radius
    # We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    # So minimize -r.
    # But -r is non-smooth.
    # We can use a soft min approximation or just rely on the optimizer's robustness.
    # Actually, L-BFGS-B can handle non-smooth functions to some extent if we provide gradients, but we don't have easy gradients for min().
    
    # Alternative: Use a custom gradient ascent loop which is more controllable.
    
    max_iter = 2000
    step_size = 0.02
    lr_decay = 0.999
    
    for i in range(max_iter):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # If we hit the boundary, we might get stuck.
        # Gradient calculation
        grad = np.zeros_like(centers)
        
        # Pair forces: repel close pairs
        # Use a threshold to only compute for close pairs to save time
        threshold = 2.0 * (current_r + 0.05)
        active_pairs = np.argwhere(dist < threshold)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        for i, j in active_pairs:
            v = centers[i] - centers[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                # Force is proportional to 1/dist or constant?
                # Constant unit vector is okay for steepest ascent of min distance
                v_norm = v / norm_v
                grad[i] += v_norm
                grad[j] -= v_norm
        
        # Wall forces: push towards center if close to wall
        active_walls = np.where(wall_dists < current_r + 0.05)[0]
        for i in active_walls:
            # Push towards origin
            norm_c = np.linalg.norm(centers[i])
            if norm_c > 1e-9:
                grad[i] -= centers[i] / norm_c
            else:
                # If at center, random push? Or just stay
                pass
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad /= grad_norm
            
        # Update
        centers += step_size * grad
        
        # Clamp to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= lr_decay
        
        # Early stopping if no improvement?
        # We can track best r.
        
    return centers
