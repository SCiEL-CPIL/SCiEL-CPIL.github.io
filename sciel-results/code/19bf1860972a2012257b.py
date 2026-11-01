import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def score(positions):
        # positions: (N, 2)
        # Calculate pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Calculate distance to boundaries
        # Square is [-0.5, 0.5]
        # dist to left: 0.5 - x (if x < 0.5), etc.
        # Actually, min(0.5 - x, 0.5 + x) for x in [-0.5, 0.5]
        # dist to boundary for point (x,y) is min(0.5-|x|, 0.5-|y|)
        dx = 0.5 - np.abs(positions[:, 0])
        dy = 0.5 - np.abs(positions[:, 1])
        min_bound_dist = np.min(np.minimum(dx, dy))
        
        # If any point is outside, the score is invalid/0.
        # We assume positions are clamped or kept inside by the optimizer.
        # But for safety, check if any |x| > 0.5 or |y| > 0.5
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
            
        return min(min_pair_dist, min_bound_dist) / 2.0 # Radius is half the min distance
        
    def optimize(init_pos, time_limit):
        pos = init_pos.copy()
        # Velocity / Step size
        step_size = 0.01
        decay = 0.999
        noise_scale = 0.01
        
        best_pos = pos.copy()
        best_score = score(pos)
        
        # Precompute indices for pairwise forces? No, vectorize.
        
        while time.time() < time_limit:
            # Calculate forces
            # Pairwise repulsion
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            # Avoid division by zero
            dist[dist < 1e-12] = 1e-12
            # Force magnitude: 1 / d^2
            # Force vector: (diff / d) * (1/d^2) = diff / d^3
            # We want to push apart.
            # F_i = sum_j (pos_i - pos_j) / |pos_i - pos_j|^3
            # Using vectorization:
            # dist_inv3 = 1 / dist**3
            # But we need to exclude diagonal.
            np.fill_diagonal(dist, np.inf)
            dist_inv3 = 1.0 / (dist**3)
            # Force from j on i is proportional to (pos_i - pos_j) / d^3
            # F_i = sum_j diff[i,j] * dist_inv3[i,j]
            forces = np.sum(diff * dist_inv3[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            # Push away from boundaries if close
            # Let's define a "soft wall" at 0.5 - margin
            margin = 0.1 # Tune this
            # For x: if x > 0.5 - margin, force is negative (push left)
            # if x < -0.5 + margin, force is positive (push right)
            # Linear spring: F = -k * (x - (0.5 - margin))
            
            # x-boundary
            x = pos[:, 0]
            fx = np.zeros_like(x)
            # Right wall
            mask_r = x > 0.5 - margin
            fx[mask_r] -= (x[mask_r] - (0.5 - margin))
            # Left wall
            mask_l = x < -0.5 + margin
            fx[mask_l] -= (x[mask_l] - (-0.5 + margin))
            
            # y-boundary
            y = pos[:, 1]
            fy = np.zeros_like(y)
            mask_r = y > 0.5 - margin
            fy[mask_r] -= (y[mask_r] - (0.5 - margin))
            mask_l = y < -0.5 + margin
            fy[mask_l] -= (y[mask_l] - (-0.5 + margin))
            
            forces[:, 0] += fx
            forces[:, 1] += fy
            
            # Update positions
            # Add noise for exploration
            noise = np.random.normal(0, noise_scale, pos.shape)
            pos = pos + step_size * forces + noise
            
            # Clamp to square to ensure validity
            pos = np.clip(pos, -0.5, 0.5)
            
            # Annealing
            step_size *= decay
            noise_scale *= decay
            
            # Check score
            current_score = score(pos)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                
            # If step size is too small, break
            if step_size < 1e-6:
                break
                
        return best_pos, best_score

    # Initialize with grid + jitter
    best_overall_pos = None
    best_overall_score = 0.0
    
    # Run multiple restarts
    num_restarts = 10 # Will be limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave 5s for final check
            break
            
        # Create initial grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        init = grid[:N].copy()
        
        # Add jitter
        init += np.random.normal(0, 0.05, init.shape)
        init = np.clip(init, -0.49, 0.49)
        
        # Allocate remaining time to this run, but ensure we don't overshoot
        # Simple: just run optimize with a tight inner loop check
        pos, sc = optimize(init, deadline - 0.5)
        
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_pos = pos
            
        # Optional: Local refinement on the best found so far
        # We can do a final polish on the best position found in all restarts
        
    if best_overall_pos is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Final polish: run optimize on the best position found with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_overall_pos, _ = optimize(best_overall_pos, deadline)
        
    return best_overall_pos
