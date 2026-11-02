import numpy as np
import time

N = 200

def construct():
    # Initialization: Square grid
    # 15x14 = 210 points. We need 200.
    # Let's generate a 14x15 grid (210) and take the first 200?
    # Or better: 14x14 = 196. Add 4 points?
    # Let's just use a 15x14 grid and select 200 points evenly.
    
    # Simple init: 14x15 grid
    m, k = 14, 15
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(k) + 0.5) / k - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # We have m*k = 210 points.
    # We need N=200.
    # Let's just take the first N points.
    positions = grid[:N].copy()
    
    # Optimization: Gradient Ascent on Soft-Min of distances
    # Objective: maximize softmin( D_ij/2, B_i )
    # Gradient of softmin(x) w.r.t x_k is softmax(-beta * x)_k
    
    beta = 20.0 # Sharpness
    step_size = 0.005
    decay = 0.995
    max_steps = 10000
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        # diff[i, j] = c_i - c_j
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask out self-distances (diagonal)
        np.fill_diagonal(dist, np.inf)
        
        # Safety for zero distances
        dist_safe = np.where(dist > 1e-8, dist, 1e-8)
        
        # Pair weights
        # w_ij = exp(-beta * dist_ij / 2)
        pair_vals = dist_safe / 2.0
        pair_weights = np.exp(-beta * pair_vals)
        pair_weights.fill_diagonal(0)
        
        # Boundary distances
        bx = 0.5 - np.abs(positions[:, 0])
        by = 0.5 - np.abs(positions[:, 1])
        
        # Weights for boundary terms
        w_bx = np.exp(-beta * bx)
        w_by = np.exp(-beta * by)
        
        # Total weight sum
        total_weight = np.sum(pair_weights) + np.sum(w_bx) + np.sum(w_by)
        
        # Coeff matrix for pair gradients
        # coeff[i,j] = w_ij / dist_ij
        coeff = pair_weights / dist_safe
        coeff[np.isinf(coeff)] = 0
        coeff[np.isnan(coeff)] = 0
        
        # Pair gradients
        sum_coeff = np.sum(coeff, axis=1, keepdims=True) # (N, 1)
        weighted_sum = coeff @ positions # (N, 2)
        
        pair_grads = (positions * sum_coeff - weighted_sum) / total_weight
        
        # Boundary gradients
        sign_x = np.sign(positions[:, 0])
        sign_y = np.sign(positions[:, 1])
        
        bnd_grads_x = (w_bx / total_weight) * (-sign_x)
        bnd_grads_y = (w_by / total_weight) * (-sign_y)
        
        grads = pair_grads
        grads[:, 0] += bnd_grads_x
        grads[:, 1] += bnd_grads_y
        
        # Update
        positions += step_size * grads
        
        # Project to square
        positions = np.clip(positions, -0.5, 0.5)
        
        # Decay step size
        step_size *= decay

    return positions
