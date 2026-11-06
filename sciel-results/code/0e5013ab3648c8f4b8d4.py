"""Improved program: vectorized gradient descent with adaptive step and multiple restarts."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    max_time = 115.0  # Leave 5s margin
    
    # Best known radius for reference
    r_best = 0.036612798904
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple restarts with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() - start_time > max_time:
            break
            
        rng = np.random.default_rng(seed)
        
        # Initialize with a perturbed grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N].copy()
        
        # Add random perturbation to break symmetry
        centers += rng.normal(0, 0.002, size=centers.shape)
        
        # Target radius: aim slightly above best known
        r_target = r_best * 1.0005
        
        # Optimization parameters
        n_steps = 50000
        lr = 0.01
        momentum = np.zeros_like(centers)
        mu = 0.85
        beta2 = 0.999  # For adaptive learning rate (Adam-like)
        m_v = np.zeros_like(centers)
        v_v = np.zeros_like(centers)
        eps = 1e-8
        
        for step in range(n_steps):
            if time.time() - start_time > max_time - 2.0:
                break
                
            # Adaptive learning rate (Adam-like)
            current_lr = lr / np.sqrt(step + 1)
            
            # Compute pairwise differences and distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Mask out diagonal
            np.fill_diagonal(dist, np.inf)
            
            # Calculate forces using smooth hinge loss
            # We want to maximize the minimum distance
            # Use a soft penalty for pairs too close
            gap = 2 * r_target - dist
            mask = gap > 0
            safe_dist = np.where(mask, dist, 1.0)
            
            # Force: push apart overlapping circles
            factor = np.zeros_like(dist)
            factor[mask] = gap[mask] / safe_dist[mask]
            
            forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Wall constraints with stronger penalty
            x = centers[:, 0]
            y = centers[:, 1]
            
            # Left wall
            gap_x_left = (r_target - 0.5) - x
            forces[:, 0] += np.where(gap_x_left > 0, gap_x_left * 10, 0)
            
            # Right wall
            gap_x_right = x - (0.5 - r_target)
            forces[:, 0] -= np.where(gap_x_right > 0, gap_x_right * 10, 0)
            
            # Bottom wall
            gap_y_bottom = (r_target - 0.5) - y
            forces[:, 1] += np.where(gap_y_bottom > 0, gap_y_bottom * 10, 0)
            
            # Top wall
            gap_y_top = y - (0.5 - r_target)
            forces[:, 1] -= np.where(gap_y_top > 0, gap_y_top * 10, 0)
            
            # Adam update
            m_v = beta2 * m_v + (1 - beta2) * forces
            v_v = beta2 * v_v + (1 - beta2) * forces * forces
            m_hat = m_v / (1 - beta2 ** (step + 1))
            v_hat = v_v / (1 - beta2 ** (step + 1))
            
            centers += -current_lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Clip to valid region
            centers = np.clip(centers, -0.5 + r_target * 0.85, 0.5 - r_target * 0.85)
        
        # Evaluate the actual radius achieved
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        x = centers[:, 0]
        y = centers[:, 1]
        min_x_dist = np.min(np.minimum(x + 0.5, 0.5 - x))
        min_y_dist = np.min(np.minimum(y + 0.5, 0.5 - y))
        min_wall_dist = min(min_x_dist, min_y_dist)
        
        achieved_radius = min(min_pair_dist / 2, min_wall_dist)
        
        if achieved_radius > best_radius:
            best_radius = achieved_radius
            best_centers = centers.copy()
    
    return best_centers if best_centers is not None else np.zeros((N, 2))
# EVOLVE-BLOCK-END
