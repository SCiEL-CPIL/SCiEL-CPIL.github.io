import numpy as np
import time
import sys

N = 100

# EVOLVE-BLOCK-START

def _soft_min_radius_and_grad(c, beta):
    """
    Computes the soft-min radius and its gradient.
    c: (N, 3) array of centers.
    beta: inverse temperature for soft-min.
    Returns:
        r_soft: float, the soft-min radius value.
        grad: (N, 3) array, the gradient of r_soft with respect to c.
    """
    N = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist = np.sqrt(dist_sq)
    # We only need upper triangle to avoid double counting, but for soft min 
    # over all pairs, we can just take the flattened upper triangle or all off-diagonals.
    # Taking all off-diagonals is simpler for indexing if we use np.where.
    i_idx, j_idx = np.where(mask)
    d_pair = dist[i_idx, j_idx] / 2.0
    
    # Boundary distances
    b_dist = 0.5 - np.abs(c)
    d_bound = b_dist.flatten()
    
    # Combine all distances
    d_all = np.concatenate([d_pair, d_bound])
    
    # Soft min calculation
    min_d = np.min(d_all)
    shifted = d_all - min_d
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    # Gradient calculation
    grad = np.zeros_like(c)
    
    # 1. Pairwise part
    w_pair = weights[:len(d_pair)]
    
    # Avoid division by zero
    safe_dist = np.where(dist[i_idx, j_idx] > 1e-12, dist[i_idx, j_idx], 1.0)
    
    c_i = c[i_idx]
    c_j = c[j_idx]
    diff_vec = c_i - c_j
    unit_vecs = diff_vec / safe_dist[:, np.newaxis]
    
    # d(dist/2)/dc_i = (c_i - c_j) / dist
    # d(dist/2)/dc_j = (c_j - c_i) / dist
    
    w_vecs = w_pair[:, np.newaxis] * unit_vecs
    
    np.add.at(grad, i_idx, w_vecs)
    np.add.at(grad, j_idx, -w_vecs)
    
    # 2. Boundary part
    w_bound = weights[len(d_pair):]
    w_bound_mat = w_bound.reshape(N, 3)
    
    # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    signs = np.sign(c)
    grad_bound = -w_bound_mat * signs
    
    grad += grad_bound
    
    r_soft = min_d - (1.0 / beta) * np.log(S)
    
    return r_soft, grad

def _hard_min_radius(c):
    """
    Computes the true maximum radius (hard min).
    """
    N = len(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    b_dist = 0.5 - np.abs(c)
    min_bound_dist = np.min(b_dist)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initialization: Jittered grid
    # A 5x5x5 grid has 125 points. We pick 100.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    np.random.seed(42)
    idx = np.random.permutation(len(grid))
    c = grid[idx[:N]].copy()
    
    # Add small jitter to break symmetry
    c += np.random.normal(0, 0.01, c.shape)
    c = np.clip(c, -0.49, 0.49)
    
    best_c = c.copy()
    best_r = _hard_min_radius(c)
    
    # Parameters
    # Start with lower beta (smoother) and increase
    beta_start = 10.0
    beta_end = 300.0
    
    # Simulated Annealing parameters
    T_start = 0.01
    T_end = 1e-5
    cooling_rate = 0.999
    
    T = T_start
    step_size = 0.01
    
    # We will iterate until time runs out
    while time.time() < deadline:
        # Update beta
        # Simple linear or exponential schedule based on iterations? 
        # Better: use time or just a fixed schedule over a large number of steps.
        # Let's use a dynamic beta that increases as we progress.
        # Since we don't know total steps, let's just keep beta high enough.
        # Or adaptively increase beta if we are stuck?
        # For now, let's use a fixed schedule based on a virtual step count.
        
        # Actually, let's just use a reasonably high beta for SA to work well on the landscape.
        # Too low beta -> too smooth, ignores close pairs. Too high -> non-smooth.
        # Let's ramp it up.
        
        # Since we don't have a fixed number of steps, let's ramp beta based on time elapsed?
        # No, better to have a fixed number of "epochs" if possible, but time is the constraint.
        # Let's just use a beta that is moderate, e.g., 50-100, and rely on SA to navigate.
        # Or ramp it up slowly.
        
        # Let's try a strategy:
        # Phase 1: SA with low beta (smooth) to find good region.
        # Phase 2: SA with higher beta (sharper) to refine.
        
        # Simple heuristic: 
        elapsed = time.time() - start_time
        progress = elapsed / 145.0
        current_beta = beta_start + (beta_end - beta_start) * progress
        
        r, grad = _soft_min_radius_and_grad(c, current_beta)
        
        # Proposal: Gradient step + Noise
        # Scale gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
        else:
            grad_dir = np.zeros_like(c)
            
        # Random perturbation
        noise = np.random.normal(0, step_size, c.shape)
        
        # Move
        # Combine gradient and noise
        # We want to move in direction of increasing r (gradient ascent)
        # But SA allows moves that decrease r.
        # Let's propose a move: c_new = c + alpha * grad_dir + noise
        # Or just c_new = c + noise, and use gradient to bias noise?
        # Common approach: c_new = c + step_size * grad_dir + random_noise
        
        c_new = c + 0.5 * step_size * grad_dir + noise
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new, _ = _soft_min_radius_and_grad(c_new, current_beta)
        
        delta_r = r_new - r
        
        # Acceptance criterion
        if delta_r > 0:
            c = c_new
        else:
            # Accept with probability exp(delta_r / T)
            # delta_r is negative
            if np.random.rand() < np.exp(delta_r / T):
                c = c_new
        
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            # If temperature is too low, we might get stuck. 
            # We can restart or just continue with gradient only?
            # For now, keep going.
            
        # Update best
        # Checking hard min every step is expensive. Check periodically.
        # Let's check every 10 steps or so.
        # Since we are in a time loop, we can just check if time allows.
        # Actually, _hard_min_radius is O(N^2). N=100, so 10k ops. Very fast.
        # Let's just check every iteration to be safe? 
        # 10k ops is negligible compared to soft_min_grad which is also O(N^2) but with more overhead.
        
        r_hard = _hard_min_radius(c)
        if r_hard > best_r:
            best_r = r_hard
            best_c = c.copy()
            
        # Adjust step size?
        # If we are accepting most moves, maybe increase step?
        # If rejecting most, decrease?
        # Simple: keep step size constant or decrease slowly.
        # step_size *= 0.999
        
    # Final polish with gradient ascent on best_c
    c = best_c
    remaining = deadline - time.time()
    if remaining > 0.5:
        beta_final = 500.0
        step_size = 0.005
        for _ in range(int(remaining * 50)): # Estimate steps
            if time.time() > deadline - 0.1:
                break
            r, grad = _soft_min_radius_and_grad(c, beta_final)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            c_new = c + step_size * (grad / grad_norm)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new, _ = _soft_min_radius_and_grad(c_new, beta_final)
            if r_new > r:
                c = c_new
            else:
                step_size *= 0.9
        
        r_hard = _hard_min_radius(c)
        if r_hard > best_r:
            best_r = r_hard
            best_c = c.copy()

    # Ensure validity
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c

# EVOLVE-BLOCK-END
