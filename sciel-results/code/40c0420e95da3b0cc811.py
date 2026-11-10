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
    # dist[i, j] = ||c_i - c_j||
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    
    # Wall distances
    # Distance to walls: 0.5 - |c_ik|
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
    Gradient: -exp(-beta * x_k) / sum(exp(-beta * x_j)) * grad x_k
    """
    centers = flat_centers.reshape(N, 3)
    
    # 1. Compute pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # x_pair = dist / 2
    x_pair = dist / 2.0
    
    # 2. Compute wall distances
    abs_c = np.abs(centers)
    x_wall = 0.5 - abs_c
    
    # Combine all x values
    # We need to be careful with memory and speed.
    # N=100, pairs ~5000, walls 300. Total ~5300 values.
    
    # Flatten x_pair (upper triangle)
    triu_indices = np.triu_indices(N, k=1)
    x_pairs_flat = x_pair[triu_indices]
    
    x_walls_flat = x_wall.flatten()
    
    x_all = np.concatenate([x_pairs_flat, x_walls_flat])
    
    # Calculate soft-min
    # To avoid overflow in exp, subtract min x
    x_min = np.min(x_all)
    exp_terms = np.exp(-beta * (x_all - x_min))
    sum_exp = np.sum(exp_terms)
    
    # Objective value
    val = (np.log(sum_exp) - beta * x_min) / beta
    
    # Gradient weights
    # w_k = -exp(-beta * (x_k - x_min)) / sum_exp
    # Note: The -beta*x_min term in exp cancels out in the ratio if we are careful, 
    # but let's stick to the formula:
    # grad f / grad x = -exp(-beta x_k) / sum(exp(-beta x_j))
    # Using shifted x:
    # grad f / grad x = -exp(-beta (x_k - x_min)) / sum(exp(-beta (x_j - x_min)))
    weights = -exp_terms / sum_exp
    
    # Initialize gradient
    grad_flat = np.zeros_like(flat_centers)
    grad_centers = grad_flat.reshape(N, 3)
    
    # 1. Accumulate pair gradients
    # For each pair (i, j), contribution to c_i is weights[ij] * grad x_ij
    # grad x_ij w.r.t c_i = 0.5 * (c_i - c_j) / dist_ij
    # grad x_ij w.r.t c_j = 0.5 * (c_j - c_i) / dist_ij
    
    # We have x_pairs_flat corresponding to triu_indices
    i_idx, j_idx = triu_indices
    w_pairs = weights[:len(x_pairs_flat)]
    
    # dist is the full matrix, we can use it
    # Avoid division by zero (should be inf on diagonal, but we took triu)
    d_pairs = dist[i_idx, j_idx]
    
    # Normalize directions
    # dir_ij = (c_i - c_j) / dist_ij
    # We need to handle dist=0 case, but they shouldn't be exactly 0 if initialized well.
    # If they are, gradient is undefined, but we can skip or set to 0.
    mask = d_pairs > 1e-12
    
    # Vectorized accumulation is tricky with sparse indices.
    # Loop over pairs is O(N^2) = 5000 iterations. In Python, this is slow.
    # We can use np.add.at
    # grad_c_i += w * 0.5 * (c_i - c_j) / d
    # grad_c_j += w * 0.5 * (c_j - c_i) / d
    
    # Let's try to vectorize the update using np.add.at
    # term_i = w_pairs[:, None] * 0.5 * (centers[i_idx] - centers[j_idx]) / d_pairs[:, None]
    # term_j = -term_i
    
    if np.any(mask):
        # Only process valid pairs
        valid_w = w_pairs[mask]
        valid_i = i_idx[mask]
        valid_j = j_idx[mask]
        valid_d = d_pairs[mask]
        
        vec_ij = centers[valid_i] - centers[valid_j]
        # grad x = 0.5 * vec / d
        grad_vec = 0.5 * vec_ij / valid_d[:, None]
        
        # Contribution to c_i: valid_w * grad_vec
        # Contribution to c_j: -valid_w * grad_vec
        
        contrib_i = valid_w[:, None] * grad_vec
        contrib_j = -valid_w[:, None] * grad_vec
        
        np.add.at(grad_centers, valid_i, contrib_i)
        np.add.at(grad_centers, valid_j, contrib_j)
        
    # 2. Accumulate wall gradients
    # x_wall = 0.5 - |c_ik|
    # grad x_wall w.r.t c_ik = -sign(c_ik)
    w_walls = weights[len(x_pairs_flat):]
    
    # Reshape w_walls to (N, 3)
    w_walls_reshaped = w_walls.reshape(N, 3)
    
    # sign(centers)
    signs = np.sign(centers)
    
    # grad_c = w_walls_reshaped * (-signs)
    grad_centers += -w_walls_reshaped * signs
    
    return val, grad_flat.flatten()

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
    
    # Random restarts
    num_restarts = 20
    rng = np.random.default_rng(42)
    
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization within cube
        # Add some jitter to grid to escape symmetry? Or just random.
        # Random uniform in [-0.45, 0.45] to avoid immediate wall contact issues?
        # Actually, bounds are handled by L-BFGS-B.
        centers = rng.uniform(-0.49, 0.49, size=(N, 3))
        
        # Optimization loop with increasing beta
        current_centers = centers.copy()
        beta = 10.0
        
        # Max 10 steps of beta increase
        for step in range(10):
            if time.time() > deadline:
                break
            
            # Define objective and gradient for current beta
            def obj_and_grad(flat_c):
                val, grad = _soft_min_objective_and_grad(flat_c, beta)
                return val, grad
            
            try:
                res = minimize(
                    obj_and_grad, 
                    current_centers.flatten(), 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                
                if res.success:
                    current_centers = res.x.reshape(N, 3)
                    
                    # Check actual radius
                    r = _compute_radius(current_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = current_centers.copy()
                        
            except Exception:
                pass
            
            # Increase beta
            beta *= 1.5
            
            # If beta gets too large, soft-min becomes very sharp and gradient might be noisy/steep
            # We can stop if improvement is negligible
            # But let's just run the schedule.
            
    # Final check
    if best_radius == 0.0:
        # Fallback to grid if something went wrong
        return best_centers
        
    return best_centers
