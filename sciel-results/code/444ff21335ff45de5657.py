"""Sphere packing: 50 equal spheres in unit cube, maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Wall distance
    wall_min = np.min(0.5 - np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)

def _soft_min_energy(centers, beta):
    """Differentiable surrogate for negative minimum distance (to walls and pairs)."""
    centers = np.asarray(centers, dtype=np.float64)
    # Wall terms: distance to each face
    # For each center, min distance to any wall
    wall_dists = np.minimum.reduce([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1],
        0.5 - centers[:, 2],
        0.5 + centers[:, 2],
    ])  # shape (N,)
    # Soft-min over all wall distances (N values)
    # We want to penalize when any wall distance is small
    # Use log-sum-exp style: -softmin(d) = logsumexp(-d/beta)/beta
    # Actually we want to maximize min distance, so minimize -min distance
    # Energy = -softmin(all relevant distances)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # All relevant distances: wall_dists (N) and pairwise (N*(N-1)/2)
    pair_dists = dist[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft-min: softmin(d) = -logsumexp(-d/beta) * beta
    # We want to minimize -softmin(d) = logsumexp(-d/beta) * beta
    # But for numerical stability, subtract max
    x = -all_dists / beta
    x_max = np.max(x)
    energy = beta * (x_max + np.log(np.sum(np.exp(x - x_max))))
    return energy

def _soft_min_gradient(centers, beta):
    """Gradient of soft-min energy w.r.t. centers."""
    centers = np.asarray(centers, dtype=np.float64)
    wall_dists = np.minimum.reduce([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1],
        0.5 - centers[:, 2],
        0.5 + centers[:, 2],
    ])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    x = -all_dists / beta
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    weights = exp_x / np.sum(exp_x)
    
    # d(energy)/d(centers) = -sum_j w_j * d(all_dists_j)/d(centers)
    # where energy = beta * logsumexp(-d/beta)
    # d(energy)/d(d_j) = beta * (-1/beta) * w_j = -w_j
    # So d(energy)/d(centers) = -sum_j w_j * d(d_j)/d(centers)
    
    grad = np.zeros_like(centers)
    
    # Wall contributions
    # d(0.5 - c_x)/d(c_x) = -1, d(0.5 + c_x)/d(c_x) = +1, etc.
    # For each center i, the 6 wall distances have specific gradients
    w_wall = weights[:N]  # weights for wall distances
    # But we need to know WHICH wall gave the min for each center
    # Actually, the soft-min handles all 6 walls per center simultaneously
    # Let me restructure: for each center, 6 wall distances
    wall_dists_full = np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1],
        0.5 - centers[:, 2],
        0.5 + centers[:, 2],
    ], axis=1)  # shape (N, 6)
    
    # Recompute with full structure for proper gradients
    # Let me redo this more carefully
    
    # Actually, let me use a simpler approach: compute gradient numerically or 
    # use a cleaner formulation
    
    # I'll use the fact that for the soft-min over all distances:
    # energy = beta * logsumexp(-d_k / beta) for all k
    # d(energy)/d(x) = -sum_k w_k * d(d_k)/d(x)
    
    # Wall gradients: for center i, wall distance to face 0 (x=-0.5): d = 0.5 - x_i
    #   d(d)/d(x_i) = -1
    # Wall distance to face 1 (x=+0.5): d = 0.5 + x_i
    #   d(d)/d(x_i) = +1
    # Similarly for y and z
    
    # I need weights per (center, wall) pair, not just per center
    # Let me recompute with full wall distance matrix
    
    wall_dists_6 = np.stack([
        0.5 - centers[:, 0],  # left x
        0.5 + centers[:, 0],  # right x
        0.5 - centers[:, 1],  # bottom y
        0.5 + centers[:, 1],  # top y
        0.5 - centers[:, 2],  # back z
        0.5 + centers[:, 2],  # front z
    ], axis=1)  # (N, 6)
    
    pair_idx_i, pair_idx_j = np.triu_indices(N, k=1)
    all_dists_full = np.concatenate([wall_dists_6.ravel(), pair_dists])
    x_full = -all_dists_full / beta
    x_max_full = np.max(x_full)
    exp_full = np.exp(x_full - x_max_full)
    w_full = exp_full / np.sum(exp_full)
    
    # Wall part: first N*6 weights
    w_wall = w_full[:N * 6].reshape(N, 6)
    w_pair = w_full[N * 6:]
    
    # Wall gradient contribution
    # For each center i, each wall face f:
    # d(d_{i,f})/d(c_i) is a 3-vector
    # Face 0 (x=-0.5): d = 0.5 - x_i, grad = (-1, 0, 0)
    # Face 1 (x=+0.5): d = 0.5 + x_i, grad = (1, 0, 0)
    # Face 2 (y=-0.5): d = 0.5 - y_i, grad = (0, -1, 0)
    # Face 3 (y=+0.5): d = 0.5 + y_i, grad = (0, 1, 0)
    # Face 4 (z=-0.5): d = 0.5 - z_i, grad = (0, 0, -1)
    # Face 5 (z=+0.5): d = 0.5 + z_i, grad = (0, 0, 1)
    
    wall_grads = np.zeros((N, 6, 3))
    wall_grads[:, 0, 0] = -1
    wall_grads[:, 1, 0] = 1
    wall_grads[:, 2, 1] = -1
    wall_grads[:, 3, 1] = 1
    wall_grads[:, 4, 2] = -1
    wall_grads[:, 5, 2] = 1
    
    # grad_wall = -sum_f w_wall[i,f] * wall_grads[i,f]
    # = -einsum('if,ifj->ij', w_wall, wall_grads)
    grad_wall = -np.einsum('if,ifj->ij', w_wall, wall_grads)
    
    # Pair gradient contribution
    # d(d_{ij})/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # d(d_{ij})/d(c_j) = (c_j - c_i) / |c_i - c_j|
    grad_pair = np.zeros_like(centers)
    for k in range(len(pair_idx_i)):
        i, j = pair_idx_i[k], pair_idx_j[k]
        d_ij = centers[i] - centers[j]
        dist_ij = np.linalg.norm(d_ij)
        if dist_ij > 1e-12:
            unit = d_ij / dist_ij
            grad_pair[i] += -w_pair[k] * unit
            grad_pair[j] += -w_pair[k] * (-unit)
    
    return grad_wall + grad_pair

def _optimize_from_start(start_centers, beta_schedule, time_limit):
    """Run annealed L-BFGS-B optimization from a starting configuration."""
    centers = start_centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    current_time = time.time()
    start_time = current_time
    
    # Anneal beta from large to small
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        # Run L-BFGS-B with current beta
        result = minimize(
            _soft_min_energy,
            centers,
            jac=_soft_min_gradient,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _soft_min_energy(centers, beta):
            centers = result.x
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
    
    # Nelder-Mead polish on the exact radius
    if time.time() - start_time < time_limit - 5:
        def neg_radius(x):
            return -_compute_radius(x.reshape(N, 3))
        
        nm_result = minimize(
            neg_radius,
            best_centers,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        r_nm = _compute_radius(nm_result.x.reshape(N, 3))
        if r_nm > best_radius:
            best_radius = r_nm
            best_centers = nm_result.x.reshape(N, 3).copy()
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    time_budget = DEADLINE - start_time
    
    # Beta schedule: anneal from coarse to fine
    beta_schedule = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    # Multiple restarts with different jitter patterns
    best_centers = None
    best_radius = 0.0
    
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 10:  # Not enough time for another restart
            break
        
        # Create jittered lattice initialization
        m = int(np.ceil(N ** (1.0 / 3.0)))  # m=4, so 4^3=64 > 50
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Shuffle and take first N
        idx = rng.permutation(len(grid))[:N]
        centers = grid[idx].copy()
        
        # Add small jitter
        jitter_scale = 0.01 * (restart + 1)
        centers += rng.uniform(-jitter_scale, jitter_scale, size=centers.shape)
        
        # Clamp to cube
        centers = np.clip(centers, -0.49, 0.49)
        
        # Allocate time for this restart
        time_per_restart = remaining / (n_restarts - restart)
        
        rc, r = _optimize_from_start(centers, beta_schedule, time_per_restart)
        
        if r > best_radius:
            best_radius = r
            best_centers = rc.copy()
        
        # Check if we've hit a good solution
        if best_radius > 0.1359:
            break
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final validation
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        # Return a valid fallback
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
