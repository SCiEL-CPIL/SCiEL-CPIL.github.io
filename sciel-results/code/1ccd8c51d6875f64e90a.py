import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 100  # Leave 20s margin for safety
    
    best_score = -1
    best_config = None
    
    # Multiple restarts to avoid local minima
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Use a random initialization
        np.random.seed(r) # Deterministic for reproducibility in evaluation, but different per restart
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        radii = np.full(N, 0.05)
        
        # Optimization parameters
        lr = 0.01
        penalty_weight = 10.0
        steps = 1000
        min_step_time = 0.5
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Compute distances and overlaps
            # d_ij is distance between i and j
            # We need gradients for centers and radii
            
            # Vectorized distance calculation
            # diff[i, j, k] = centers[i, k] - centers[j, k]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf) # Avoid self-interaction
            
            # Overlaps: r_i + r_j - d_ij
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlap = np.maximum(0, r_sum - dist)
            
            # Boundary violations
            # b_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # violation_i = max(0, r_i - b_i)
            b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                           np.minimum(centers[:, 1], 1 - centers[:, 1]))
            b_viol = np.maximum(0, radii - b)
            
            # Penalty
            overlap_penalty = np.sum(overlap**2)
            b_penalty = np.sum(b_viol**2)
            total_penalty = overlap_penalty + b_penalty
            
            # If no penalty, we can just increase radii
            # But we need gradients to know how to move
            
            # Gradients
            # d(overlap^2)/d(centers) and d(overlap^2)/d(radii)
            
            # Only compute gradients for pairs with overlap > 0 to save time?
            # For N=26, full matrix is 26x26, which is small.
            
            grad_centers = np.zeros_like(centers)
            grad_radii = np.zeros_like(radii)
            
            # Objective gradient: d(sum r)/dr = 1
            grad_radii += 1.0
            
            # Penalty gradients
            # P = sum_{i<j} max(0, r_i+r_j-d_ij)^2
            # dP/dr_k = 2 * sum_{j != k} overlap_kj * sign(overlap_kj)
            # dP/dx_i = 2 * sum_{j != i} overlap_ij * sign(overlap_ij) * (x_i - x_j) / d_ij
            
            # Vectorized gradient for radii
            # overlap is N x N.
            # grad_radii from overlap: 2 * sum_j overlap_ij
            # Note: overlap is symmetric.
            grad_radii_overlap = 2.0 * np.sum(overlap, axis=1)
            grad_radii -= penalty_weight * grad_radii_overlap
            
            # Boundary gradient
            # d(b_viol^2)/dr_i = 2 * b_viol_i
            # d(b_viol^2)/dx_i depends on which boundary is active
            grad_radii_b = 2.0 * b_viol
            grad_radii -= penalty_weight * grad_radii_b
            
            # Centers gradient from overlap
            # For each i, sum over j where overlap > 0
            # term = 2 * overlap_ij * (diff_ij / dist_ij)
            # diff_ij = centers[i] - centers[j]
            # grad_centers[i] -= penalty_weight * sum_j term
            
            # Avoid division by zero
            dist_safe = np.where(dist > 1e-6, dist, 1e-6)
            unit_vec = diff / dist_safe[:, :, np.newaxis]
            
            # Gradient contribution from each pair j to i
            # d(overlap^2)/d(centers[i]) = 2 * overlap * (centers[i] - centers[j]) / dist
            # We sum over j
            grad_centers_overlap = 2.0 * np.sum(overlap[:, :, np.newaxis] * unit_vec, axis=1)
            grad_centers -= penalty_weight * grad_centers_overlap
            
            # Centers gradient from boundary
            # b_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # If x_i < 1-x_i and x_i < y_i and x_i < 1-y_i, then b_i = x_i
            # d(b_viol^2)/dx_i = 2 * b_viol * db/dx_i
            # db/dx_i is 1 if x_i is the min, -1 if 1-x_i is the min, 0 otherwise.
            
            # Determine active boundary for each i
            # b_idx: 0 for x, 1 for 1-x, 2 for y, 3 for 1-y
            # We can just check which one is minimum
            x_min = centers[:, 0]
            x_max = 1 - centers[:, 0]
            y_min = centers[:, 1]
            y_max = 1 - centers[:, 1]
            
            # db/dx
            db_dx = np.zeros(N)
            db_dy = np.zeros(N)
            
            # If x_min is the min, db/dx = 1
            # If x_max is the min, db/dx = -1
            # Similarly for y
            
            # Find min index
            b_vals = np.stack([x_min, x_max, y_min, y_max], axis=1)
            min_idx = np.argmin(b_vals, axis=1)
            
            for i in range(N):
                if min_idx[i] == 0: # x
                    db_dx[i] = 1.0
                elif min_idx[i] == 1: # 1-x
                    db_dx[i] = -1.0
                elif min_idx[i] == 2: # y
                    db_dy[i] = 1.0
                elif min_idx[i] == 3: # 1-y
                    db_dy[i] = -1.0
            
            grad_centers_b = 2.0 * b_viol * np.stack([db_dx, db_dy], axis=1)
            grad_centers -= penalty_weight * grad_centers_b
            
            # Update
            centers += lr * grad_centers
            radii += lr * grad_radii
            
            # Project to ensure radii > 0 and centers in [0,1] roughly
            radii = np.maximum(radii, 0.001)
            # Don't project centers strictly, let penalty handle it, but clamp to avoid explosion
            centers = np.clip(centers, 0.0, 1.0)
            
            # Adaptive learning rate? Maybe reduce if penalty increases
            # Simple: keep constant
            
        # End of steps
        
        # Check validity and score
        # Re-calculate distances and overlaps
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, r_sum - dist)
        max_overlap = np.max(overlap)
        
        b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
        b_viol = np.maximum(0, radii - b)
        max_b_viol = np.max(b_viol)
        
        # Valid if max overlap and max boundary violation are very small
        if max_overlap < 1e-5 and max_b_viol < 1e-5:
            # Fix any tiny numerical errors by shrinking radii slightly
            # Or just accept if close enough
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_config = np.column_stack([centers, radii])
                
        # If not valid, we could try to shrink radii to make it valid
        # But the gradient ascent should have pushed it to be valid if possible.
        
    # If no valid config found, return a trivial valid one
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
        
    return best_config
