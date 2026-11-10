"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the largest common radius for given centers."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _smooth_radius_surrogate(c, beta=50.0):
    """
    Smooth surrogate for the min radius using log-sum-exp.
    Returns (smooth_min_pair/2, smooth_min_wall) and their gradients w.r.t. c.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances
    log_dists = np.log(np.maximum(dists, 1e-15))
    # Use log-sum-exp trick for soft-min: -logsumexp(-log(dists))
    max_ld = np.max(log_dists)
    soft_min_pair = -max_ld - np.log(np.sum(np.exp(log_dists - max_ld)))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    log_walls = np.log(np.maximum(wall_dists, 1e-15))
    max_lw = np.max(log_walls)
    soft_min_wall = -max_lw - np.log(np.sum(np.exp(log_walls - max_lw)))
    
    # The overall soft-min
    log_vals = np.array([soft_min_pair, soft_min_wall])
    max_lv = np.max(log_vals)
    soft_min_total = -max_lv - np.log(np.sum(np.exp(log_vals - max_lv)))
    
    return 0.5 * soft_min_total

def _objective_and_grad(c, beta=50.0):
    """
    Objective: maximize min radius. We minimize the negative.
    Using smooth approximation via log-sum-exp for soft-min.
    Returns (value, grad) where value = -smooth_radius, grad = d(value)/dc
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances with temperature 1/beta
    # soft_min(x) = (1/beta) * log(sum(exp(-beta * x)))
    # gradient: -exp(-beta*x) / sum(exp(-beta*x))
    
    # For pairwise:
    neg_beta_dists = -beta * dists
    max_nd = np.max(neg_beta_dists)
    exp_nd = np.exp(neg_beta_dists - max_nd)
    sum_exp = np.sum(exp_nd)
    weights = exp_nd / sum_exp  # shape (n, n)
    
    # soft_min_pair = (1/beta) * (max_nd + log(sum_exp))
    soft_min_pair = (1.0 / beta) * (max_nd + np.log(sum_exp))
    
    # Gradient of soft_min_pair w.r.t. c_i:
    # d(soft_min_pair)/dc_i = sum_j weights[i,j] * d(-d_ij)/dc_i
    # d(d_ij)/dc_i = (c_i - c_j) / d_ij
    # So d(soft_min_pair)/dc_i = -sum_j weights[i,j] * (c_i - c_j) / d_ij
    
    # Build gradient
    grad_pair = np.zeros_like(c)
    for i in range(n):
        for j in range(n):
            if i != j and weights[i, j] > 1e-15:
                d_ij = dists[i, j]
                if d_ij > 1e-10:
                    grad_pair[i] -= weights[i, j] * (c[i] - c[j]) / d_ij
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    neg_beta_walls = -beta * wall_dists
    max_nw = np.max(neg_beta_walls)
    exp_nw = np.exp(neg_beta_walls - max_nw)
    sum_exp_w = np.sum(exp_nw)
    weights_w = exp_nw / sum_exp_w
    
    soft_min_wall = (1.0 / beta) * (max_nw + np.log(sum_exp_w))
    
    # Gradient of soft_min_wall w.r.t. c:
    # d(soft_min_wall)/dc_ik = weights_w[i,k] * d(-wall_dist_ik)/dc_ik
    # wall_dist_ik = 0.5 - |c_ik|
    # d(wall_dist_ik)/dc_ik = -sign(c_ik)
    # So d(soft_min_wall)/dc_ik = weights_w[i,k] * sign(c_ik)
    
    grad_wall = np.zeros_like(c)
    for i in range(n):
        for k in range(3):
            if weights_w[i, k] > 1e-15:
                grad_wall[i, k] = weights_w[i, k] * np.sign(c[i, k])
    
    # Overall soft-min of (soft_min_pair, soft_min_wall)
    # Actually, the true radius is min(soft_min_pair, soft_min_wall) / 2
    # Let's just combine them with another soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    neg_beta_vals = -beta * vals
    max_nv = np.max(neg_beta_vals)
    exp_nv = np.exp(neg_beta_vals - max_nv)
    sum_exp_v = np.sum(exp_nv)
    w_pair = exp_nv[0] / sum_exp_v
    w_wall = exp_nv[1] / sum_exp_v
    
    soft_min_total = (1.0 / beta) * (max_nv + np.log(sum_exp_v))
    
    value = -0.5 * soft_min_total
    grad = 0.5 * (w_pair * grad_pair + w_wall * grad_wall)
    
    return value, grad

def _refine_force_directed(c, time_limit=100.0):
    """
    Force-directed refinement: push spheres apart and toward center.
    Uses actual (non-smooth) radius for acceptance.
    """
    c = np.array(c, dtype=np.float64)
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    step_size = 0.005
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                else:
                    # Repulsive force proportional to 1/dist^2
                    f = d / (dist * dist)
                    forces[i] += f
                    forces[j] -= f
        
        # Wall attraction (push away from walls)
        for i in range(n):
            for k in range(3):
                dist_to_wall = 0.5 - np.abs(c[i, k])
                if dist_to_wall < 0.05:
                    forces[i, k] += np.sign(c[i, k]) * (0.05 - dist_to_wall) * 10.0
        
        # Normalize forces
        norm_f = np.linalg.norm(forces, axis=1)
        for i in range(n):
            if norm_f[i] > 1.0:
                forces[i] /= norm_f[i]
        
        # Move
        new_c = c + step_size * forces
        
        # Clamp to cube
        new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check if improved
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            c = new_c
            best_c = c.copy()
            best_r = new_r
            step_size = min(step_size * 1.05, 0.02)
        else:
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses force-directed optimization with time limit.
    """
    total_time_limit = 120.0
    start_time = time.time()
    
    # Initial configuration: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Multiple restarts with different initial perturbations
    n_restarts = 5
    restart_budget = (total_time_limit - 30.0) / n_restarts  # Leave 30s for final
    
    for r in range(n_restarts):
        if time.time() - start_time > total_time_limit - 20.0:
            break
        
        # Perturb initial configuration
        if r == 0:
            c_init = c.copy()
        else:
            c_init = c.copy() + np.random.randn(N, 3) * 0.02
            c_init = np.clip(c_init, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Refine
        time_left = min(restart_budget, total_time_limit - 25.0 - (time.time() - start_time))
        if time_left < 5.0:
            break
        
        c_refined, r_refined = _refine_force_directed(c_init, time_limit=time_left)
        
        if r_refined > best_r:
            best_c = c_refined
            best_r = r_refined
    
    # Final high-precision polish
    time_left = total_time_limit - 5.0 - (time.time() - start_time)
    if time_left > 10.0:
        c_final, r_final = _refine_force_directed(best_c, time_limit=time_left)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final
    
    # Ensure validity: clip to cube interior
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_c
# EVOLVE-BLOCK-END
