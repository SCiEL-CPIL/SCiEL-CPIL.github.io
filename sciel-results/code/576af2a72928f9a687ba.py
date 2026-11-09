import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize with grid
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.5/side, 1 - 0.5/side, side)
    ys = np.linspace(0.5/side, 1 - 0.5/side, side)
    
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([xs[i], ys[j]])
                
    positions = np.array(positions[:N], dtype=float)
    radii = np.full(N, 0.05, dtype=float)
    
    # Optimization parameters
    lr_pos = 0.01
    lr_rad = 0.001
    lambda_overlap = 10.0
    lambda_boundary = 10.0
    steps = 5000
    
    best_score = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Compute distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        dist[dist == 0] = 1e-6
        
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap
        overlap = np.maximum(sum_r - dist, 0)
        
        # Boundary violation
        vx = np.maximum(0, radii - positions[:, 0]) + np.maximum(0, radii - (1 - positions[:, 0]))
        vy = np.maximum(0, radii - positions[:, 1]) + np.maximum(0, radii - (1 - positions[:, 1]))
        
        # Loss (not directly used for update, but for tracking)
        # loss = -np.sum(radii) + lambda_overlap * np.sum(overlap**2) + lambda_boundary * np.sum(vx**2 + vy**2)
        
        # Gradients
        grad_pos = np.zeros_like(positions)
        
        # Vectorized gradient for overlap
        safe_dist = dist.copy()
        safe_dist[safe_dist == 0] = 1e-6
        
        mask = overlap > 0
        term = np.zeros_like(diff)
        term[mask] = 2 * lambda_overlap * overlap[mask] * diff[mask] / safe_dist[mask, np.newaxis]
        
        grad_pos += np.sum(term, axis=1)
        
        # Boundary gradient
        grad_boundary = np.zeros_like(positions)
        
        mask_l = positions[:, 0] < radii
        grad_boundary[mask_l, 0] -= 2 * lambda_boundary * (radii[mask_l] - positions[mask_l, 0])
        mask_r = positions[:, 0] > 1 - radii
        grad_boundary[mask_r, 0] += 2 * lambda_boundary * (radii[mask_r] - (1 - positions[mask_r, 0]))
        mask_b = positions[:, 1] < radii
        grad_boundary[mask_b, 1] -= 2 * lambda_boundary * (radii[mask_b] - positions[mask_b, 1])
        mask_t = positions[:, 1] > 1 - radii
        grad_boundary[mask_t, 1] += 2 * lambda_boundary * (radii[mask_t] - (1 - positions[mask_t, 1]))
        
        grad_pos += grad_boundary
        
        # dLoss/drad
        grad_rad = -np.ones(N)
        grad_rad += 2 * lambda_overlap * np.sum(overlap, axis=1)
        
        # Boundary gradient for radii
        grad_rad_boundary = np.zeros(N)
        mask_l = positions[:, 0] < radii
        grad_rad_boundary[mask_l] += 2 * lambda_boundary * (radii[mask_l] - positions[mask_l, 0])
        mask_r = positions[:, 0] > 1 - radii
        grad_rad_boundary[mask_r] += 2 * lambda_boundary * (radii[mask_r] - (1 - positions[mask_r, 0]))
        mask_b = positions[:, 1] < radii
        grad_rad_boundary[mask_b] += 2 * lambda_boundary * (radii[mask_b] - positions[mask_b, 1])
        mask_t = positions[:, 1] > 1 - radii
        grad_rad_boundary[mask_t] += 2 * lambda_boundary * (radii[mask_t] - (1 - positions[mask_t, 1]))
        
        grad_rad += grad_rad_boundary
        
        # Update
        positions -= lr_pos * grad_pos
        radii -= lr_rad * grad_rad
        
        # Clamp radii to positive
        radii = np.maximum(radii, 0.01)
        
        # Clamp positions to [0, 1]
        positions = np.clip(positions, 0.0, 1.0)
        
        # Track best
        current_score = np.sum(radii)
        # Check validity
        max_overlap = np.max(overlap)
        max_violation = np.max(vx) + np.max(vy)
        
        if max_overlap < 1e-5 and max_violation < 1e-5:
            if current_score > best_score:
                best_score = current_score
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Increase lambda to enforce constraints more strictly
        lambda_overlap *= 1.01
        lambda_boundary *= 1.01
        # Decrease learning rate to converge
        lr_pos *= 0.99
        lr_rad *= 0.99
        
    result = np.column_stack((best_positions, best_radii))
    return result
