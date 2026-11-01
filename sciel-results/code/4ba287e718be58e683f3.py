import numpy as np

def construct():
    N = 50
    # Initialize with a 4x4x4 grid subset
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Use a fixed seed for the random selection to ensure reproducibility of the "good" start,
    # but the optimization itself will be deterministic given the start.
    # Actually, let's try a few random starts and keep the best one?
    # Given 120s, we can do 3-5 restarts.
    
    best_centers = None
    best_score = -1.0
    
    # Run optimization multiple times with different random inits
    for run in range(5):
        np.random.seed(42 + run)
        idx = np.random.permutation(64)[:50]
        centers = grid[idx].copy()
        
        # Optimization Parameters
        max_iter = 15000
        step_size = 0.005
        learning_rate_decay = 0.9995
        temperature = 0.05
        temp_decay = 0.999
        eps = 1e-9
        
        for i in range(max_iter):
            # Compute differences and distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Wall distances
            abs_centers = np.abs(centers)
            max_coord = np.max(abs_centers, axis=1)
            wall_dists = 0.5 - max_coord
            
            current_r = min(np.min(dist) / 2.0, np.min(wall_dists))
            
            # Determine active constraints with a "soft" threshold
            threshold_pair = 2.0 * current_r * 1.5
            threshold_wall = current_r * 1.5
            
            # Vectorized Gradient Calculation
            grad = np.zeros_like(centers)
            
            # Pair contributions
            active_mask = dist < threshold_pair
            safe_dist = np.where(active_mask, dist, 1.0)
            unit_diff = diff / safe_dist[:, :, np.newaxis]
            weight = active_mask.astype(float)
            grad += np.sum(unit_diff * weight[:, :, np.newaxis], axis=1)
            
            # Wall contributions
            active_walls = wall_dists < threshold_wall
            limiting_coords = np.argmax(abs_centers, axis=1)
            one_hot = np.zeros((N, 3))
            one_hot[np.arange(N), limiting_coords] = 1.0
            wall_grad = -np.sign(centers) * one_hot
            grad += wall_grad * active_walls[:, np.newaxis]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                grad /= grad_norm
            else:
                grad = np.random.randn(N, 3) * 0.1
            
            # Add noise
            noise = np.random.randn(N, 3) * temperature
            
            # Update
            centers += (step_size * grad) + noise
            
            # Clamp to strictly inside cube
            centers = np.clip(centers, -0.5 + eps, 0.5 - eps)
            
            # Decay
            step_size *= learning_rate_decay
            temperature *= temp_decay
        
        # Calculate final score
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        min_wall_dist = np.min(0.5 - max_coord)
        
        score = min(min_pair_dist / 2.0, min_wall_dist)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    return best_centers
