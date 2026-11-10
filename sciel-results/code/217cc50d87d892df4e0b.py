import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Compute the actual radius for a given set of centers.
    centers: (N, 3) array
    """
    if centers.shape != (N, 3):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    
    # Wall distances
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    if not np.isfinite(r) or r < 0:
        return 0.0
        
    return r

def _soft_min_objective_and_grad(flat_centers, beta):
    """
    Objective: Minimize (1/beta) * log(sum(exp(-beta * x_k)))
    where x_k are the relevant distances (pair/2 and wall).
    """
    centers = flat_centers.reshape(N, 3)
    
    # 1. Compute pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    x_pair = dist / 2.0
    
    # 2. Compute wall distances
    abs_c = np.abs(centers)
    x_wall = 0.5 - abs_c
    
    # Flatten x_pair (upper triangle)
    triu_indices = np.triu_indices(N, k=1)
    x_pairs_flat = x_pair[triu_indices]
    x_walls_flat = x_wall.flatten()
    
    x_all = np.concatenate([x_pairs_flat, x_walls_flat])
    
    # Calculate soft-min
    x_min = np.min(x_all)
    exp_terms = np.exp(-beta * (x_all - x_min))
    sum_exp = np.sum(exp_terms)
    
    val = (np.log(sum_exp) - beta * x_min) / beta
    
    weights = -exp_terms / sum_exp
    
    grad_flat = np.zeros_like(flat_centers)
    grad_centers = grad_flat.reshape(N, 3)
    
    # 1. Accumulate pair gradients
    i_idx, j_idx = triu_indices
    w_pairs = weights[:len(x_pairs_flat)]
    d_pairs = dist[i_idx, j_idx]
    
    mask = d_pairs > 1e-12
    
    if np.any(mask):
        valid_w = w_pairs[mask]
        valid_i = i_idx[mask]
        valid_j = j_idx[mask]
        valid_d = d_pairs[mask]
        
        vec_ij = centers[valid_i] - centers[valid_j]
        grad_vec = 0.5 * vec_ij / valid_d[:, None]
        
        contrib_i = valid_w[:, None] * grad_vec
        contrib_j = -valid_w[:, None] * grad_vec
        
        np.add.at(grad_centers, valid_i, contrib_i)
        np.add.at(grad_centers, valid_j, contrib_j)
        
    # 2. Accumulate wall gradients
    w_walls = weights[len(x_pairs_flat):]
    w_walls_reshaped = w_walls.reshape(N, 3)
    signs = np.sign(centers)
    grad_centers += -w_walls_reshaped * signs
    
    return val, grad_flat.flatten()

def _md_step(centers, alpha, T, rng, p=4):
    """
    Perform one Molecular Dynamics step.
    """
    N = len(centers)
    # Pairwise forces
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist2, 1.0) # Avoid division by zero
    
    # Force ~ r / r^(p+2) = r / r^p ? No.
    # Potential U ~ 1/r^p. Force = -grad U ~ r / r^(p+2) = r / r^p.
    # Let's use U ~ 1/r^2. Force ~ r / r^4 = r / r^4.
    # Or simpler: Force = diff / dist2. (Coulomb-like)
    # This is strong repulsion.
    
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    force = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Wall forces
    # Push away from walls if close.
    # Use a soft wall potential: U_wall ~ 1/(0.5 - |c|)^2
    # Force ~ 1/(0.5 - |c|)^3 * sign(c)
    
    abs_c = np.abs(centers)
    dist_to_wall = 0.5 - abs_c
    # Avoid singularity
    dist_to_wall_safe = np.maximum(dist_to_wall, 0.01)
    
    # Wall force magnitude
    wall_force_mag = 1.0 / (dist_to_wall_safe ** 3)
    # Direction: away from wall, so towards center.
    # If c > 0, force is negative. If c < 0, force is positive.
    # So force = -sign(c) * mag
    wall_force = -np.sign(centers) * wall_force_mag
    
    total_force = force + wall_force
    
    # Update
    noise = rng.normal(0, np.sqrt(2 * T), size=centers.shape)
    new_centers = centers + alpha * total_force + noise
    
    # Clip to bounds
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers

def _anneal(initial_centers, rng, time_limit):
    """
    Run Simulated Annealing / MD.
    """
    centers = initial_centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    T = 0.01
    alpha = 0.001
    cooling_rate = 0.99
    steps_per_iter = 100
    
    start_time = time.time()
    
    while time.time() < time_limit:
        for _ in range(steps_per_iter):
            centers = _md_step(centers, alpha, T, rng)
            
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        T *= cooling_rate
        # alpha might need to decrease or stay constant. 
        # As T decreases, we want smaller steps to settle.
        alpha *= 0.999
        
        if T < 1e-6:
            break
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial best: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(42)
    
    num_restarts = 30
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        # Mix of random and jittered grid
        if r_idx % 2 == 0:
            # Jittered grid
            jitter = rng.uniform(-0.05, 0.05, size=(N, 3))
            centers = best_centers + jitter
            centers = np.clip(centers, -0.49, 0.49)
        else:
            # Random
            centers = rng.uniform(-0.49, 0.49, size=(N, 3))
            
        # MD Annealing
        # Limit time per restart to ~2s
        restart_deadline = time.time() + 2.0
        if restart_deadline > deadline:
            restart_deadline = deadline
            
        try:
            md_centers, md_r = _anneal(centers, rng, restart_deadline)
            
            if md_r > best_radius:
                best_radius = md_r
                best_centers = md_centers.copy()
                
            # Polish with L-BFGS-B
            # Only if we have time
            if time.time() < deadline - 5.0:
                current_centers = md_centers.copy()
                beta = 20.0
                
                def obj_and_grad(flat_c):
                    val, grad = _soft_min_objective_and_grad(flat_c, beta)
                    return val, grad
                
                res = minimize(
                    obj_and_grad, 
                    current_centers.flatten(), 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success:
                    polished_centers = res.x.reshape(N, 3)
                    r = _compute_radius(polished_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = polished_centers.copy()
                        
        except Exception:
            pass
            
    return best_centers
