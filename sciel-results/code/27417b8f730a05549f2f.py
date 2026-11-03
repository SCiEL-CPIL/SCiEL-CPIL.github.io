import numpy as np
import time
import math

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    Returns (score, min_pair_dist, min_bound_dist, critical_pairs, critical_points)
    """
    # Pairwise distances
    # Using broadcasting is memory heavy for N=300 (300x300x2 = 180k floats, ~1.4MB, ok)
    # But computing all pairs every step in SA might be slow.
    # For SA, we can just compute the score.
    
    # Boundary
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Pairwise
    # To save time, we can just compute the min distance.
    # If we need gradients later, we recompute.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    
    # Identify critical constraints for potential gradient step
    # We need indices for the polish phase
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    
    # Critical pairs are those with dist < score + epsilon
    eps = 1e-4
    critical_mask = dists < (score + eps)
    # Get indices of critical pairs
    # We need to map back to i_idx, j_idx
    # It's easier to just return the matrices or recompute in polish.
    # For SA, we just need the score.
    
    return score

def polish(points, time_budget):
    """
    Gradient-based polish with two-phase soft-min: moderate k first, then sharp k.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Two-phase: phase 0 moderate k, phase 1 sharp k
    k_phase0 = 50.0
    k_phase1 = 500.0
    lr_phase0 = 0.003
    lr_phase1 = 1e-5
    
    phase0_time = time_budget * 0.6
    phase1_time = time_budget * 0.4
    
    # Phase 0: moderate k for broad improvement
    k = k_phase0
    lr = lr_phase0
    phase_end = start_time + phase0_time
    
    while time.time() < phase_end:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        shifted_dists = all_dists - current_score
        mask_close = shifted_dists < 0.05
        if not np.any(mask_close):
            break
            
        close_dists = shifted_dists[mask_close]
        exp_weights = np.exp(-k * close_dists)
        sum_weights = np.sum(exp_weights)
        
        if sum_weights == 0:
            break
            
        weights_full = np.zeros_like(all_dists)
        weights_full[mask_close] = exp_weights / sum_weights
        
        grads = np.zeros_like(points)
        
        pair_weights = weights_full[:num_pairs]
        d_pair = dists_matrix[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]
            w_ij = pair_weights[mask_nonzero, np.newaxis]
            forces = np.zeros((N, 2))
            np.add.at(forces, i_idx[mask_nonzero], v_ij * w_ij)
            np.add.at(forces, j_idx[mask_nonzero], -v_ij * w_ij)
            grads += forces
            
        bound_weights = weights_full[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-9:
            grads /= grad_norm
            points += lr * grads
            
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        lr *= 0.998
        k *= 1.005
    
    # Phase 1: sharp k for fine refinement
    k = k_phase1
    lr = lr_phase1
    phase_end = start_time + time_budget
    
    while time.time() < phase_end:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        shifted_dists = all_dists - current_score
        mask_close = shifted_dists < 0.02
        if not np.any(mask_close):
            break
            
        close_dists = shifted_dists[mask_close]
        exp_weights = np.exp(-k * close_dists)
        sum_weights = np.sum(exp_weights)
        
        if sum_weights == 0:
            break
            
        weights_full = np.zeros_like(all_dists)
        weights_full[mask_close] = exp_weights / sum_weights
        
        grads = np.zeros_like(points)
        
        pair_weights = weights_full[:num_pairs]
        d_pair = dists_matrix[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]
            w_ij = pair_weights[mask_nonzero, np.newaxis]
            forces = np.zeros((N, 2))
            np.add.at(forces, i_idx[mask_nonzero], v_ij * w_ij)
            np.add.at(forces, j_idx[mask_nonzero], -v_ij * w_ij)
            grads += forces
            
        bound_weights = weights_full[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-9:
            grads /= grad_norm
            points += lr * grads
            
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        lr *= 0.999
        k *= 1.002
    
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization
    # Generate a hexagonal lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.9
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Simulated Annealing Phase
    # We use a multi-start approach with SA.
    # Given time constraints, we will run a few restarts.
    # Each restart: Perturb current best, run SA.
    
    num_restarts = 4
    time_per_restart = (time_limit - (time.time() - start_time)) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Initialize for this restart
        if r == 0:
            current_points = best_points.copy()
        else:
            # Perturb best points to escape local minima
            current_points = best_points + np.random.normal(0, 0.05, size=best_points.shape)
            # Project inside
            rad = np.linalg.norm(current_points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                current_points[mask] /= rad[mask, np.newaxis]
                
        current_score = compute_score(current_points)
        
        # SA Parameters
        T_start = 0.04
        T_end = 0.001
        alpha = 0.997
        T = T_start
        max_sa_iters = 3000
        
        sa_start_time = time.time()
        sa_time_limit = time_per_restart * 0.55 # Leave more time for polish
        
        for it in range(max_sa_iters):
            if time.time() - sa_start_time > sa_time_limit:
                break
                
            # Pick a random point to move
            idx = np.random.randint(N)
            old_point = current_points[idx].copy()
            
            # Random move with adaptive size
            move_size = max(0.001, T * 0.4)
            new_point = old_point + np.random.normal(0, move_size, 2)
            
            # Check if inside container
            if np.linalg.norm(new_point) >= 1.0:
                continue
                
            current_points[idx] = new_point
            new_score = compute_score(current_points)
            
            # Acceptance
            if new_score > current_score:
                current_score = new_score
            else:
                delta = new_score - current_score
                if np.random.rand() < np.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[idx] = old_point
                    
            T *= alpha
            
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # 3. Polish Phase for this candidate
        # Use remaining time for this restart
        polish_time = time_per_restart * 0.4
        polished_points, polished_score = polish(current_points, polish_time)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # Final global polish on the best found
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        final_points, final_score = polish(best_points, remaining_time)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
            
    return best_points
