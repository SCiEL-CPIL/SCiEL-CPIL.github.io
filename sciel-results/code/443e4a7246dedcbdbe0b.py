import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    if r_pair < r_wall:
        return r_pair, 'pair', min_pair_idx, r_pair, r_wall
    else:
        return r_wall, 'wall', min_wall_idx, r_pair, r_wall

def soft_min_all_objective(flat_pos, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp soft-min over ALL constraints.
    We minimize J(x) = - soft_min({d_ij/2, w_ik}).
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_k)))
    So J(x) = 1/beta * log(sum(exp(-beta * x_k)))
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Flatten constraints into a single vector for LSE
    # x_k values:
    # 1. d_ij / 2 for i < j
    # 2. w_ik for all i, k
    
    # Extract upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Flatten wall distances
    wall_dists = d_walls.flatten()
    
    # Combine all constraint values
    # x = [pair_dists, wall_dists]
    x = np.concatenate([pair_dists, wall_dists])
    
    # Compute soft min
    # For numerical stability, shift by min(x)
    min_x = np.min(x)
    if min_x < -1000:
        # Fallback if values are extremely negative (shouldn't happen for distances)
        return -min_x, None
        
    exp_terms = np.exp(-beta * (x - min_x))
    log_sum_exp = np.log(np.sum(exp_terms))
    
    soft_min_val = min_x - log_sum_exp / beta
    obj = -soft_min_val
    
    # Compute gradient
    # d(obj)/d(pos) = - d(soft_min)/d(pos)
    # d(soft_min)/d(x_k) = - exp(-beta(x_k - min_x)) / sum(exp(...)) = - w_k / W
    # where w_k = exp(-beta(x_k - min_x)) and W = sum(w_k)
    # d(soft_min)/d(pos) = sum_k (d(soft_min)/d(x_k)) * (d(x_k)/d(pos))
    #                     = - (1/W) * sum_k w_k * grad_x_k
    
    # We want grad_obj = - grad_soft_min = (1/W) * sum_k w_k * grad_x_k
    
    weights = exp_terms / np.sum(exp_terms)
    
    grad_pos = np.zeros_like(pos)
    
    # 1. Pair contributions
    # x_k = d_ij / 2
    # grad_x_k wrt p_i = (p_i - p_j) / (2 * d_ij)
    # grad_x_k wrt p_j = (p_j - p_i) / (2 * d_ij) = - grad_x_k wrt p_i
    
    d_raw = dists[i_idx, j_idx]
    # Avoid division by zero
    mask = d_raw > 1e-12
    if np.any(mask):
        # Weights for pairs
        w_pairs = weights[:len(pair_dists)]
        
        # Compute vectors
        # diff[i, j] = pos[i] - pos[j]
        # We need (pos[i] - pos[j]) / d_ij for each pair
        # Efficient way:
        # grad_pos[i] += sum_{j>i} w_k * (p_i - p_j) / (2 * d_ij)
        # grad_pos[j] -= sum_{i<j} w_k * (p_i - p_j) / (2 * d_ij)
        
        # Vectorized accumulation is tricky with sparse indices.
        # Given N=100, ~5000 pairs, a loop is acceptable in Python if optimized?
        # Or use np.add.at
        
        # Let's use np.add.at for efficiency
        vecs = diff[i_idx, j_idx] # shape (num_pairs, 3)
        # Normalize
        inv_d = np.zeros_like(d_raw)
        inv_d[mask] = 1.0 / d_raw[mask]
        # Scale by 0.5
        vecs_scaled = vecs * (inv_d[:, np.newaxis] * 0.5)
        
        # Weighted sum
        # w_pairs shape (num_pairs,)
        # vecs_scaled shape (num_pairs, 3)
        # We want to add w_pairs[:, None] * vecs_scaled to grad_pos[i_idx]
        # and subtract from grad_pos[j_idx]
        
        weighted_vecs = vecs_scaled * w_pairs[:, np.newaxis]
        
        # Add to i_idx
        np.add.at(grad_pos, i_idx, weighted_vecs)
        # Subtract from j_idx
        np.add.at(grad_pos, j_idx, -weighted_vecs)
        
    # 2. Wall contributions
    # x_k = 0.5 - |p_ik|
    # grad_x_k wrt p_ik = -sign(p_ik)
    
    w_walls = weights[len(pair_dists):]
    
    # d_walls shape (N, 3)
    # pos shape (N, 3)
    signs = np.sign(pos)
    # Handle zero
    signs[signs == 0] = 1.0
    
    # grad_x = -signs
    # Contribution to grad_pos: w_walls * (-signs)
    # w_walls is flattened (N*3,)
    w_walls_reshaped = w_walls.reshape(N, 3)
    
    grad_pos += -w_walls_reshaped * signs
    
    return obj, grad_pos.flatten()

def objective_and_grad(flat_pos):
    """Original hard-min objective with analytical gradient."""
    pos = flat_pos.reshape(N, 3)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    grad = np.zeros_like(pos)
    
    if r_pair <= r_wall:
        d = min_pair_val
        if d > 1e-12:
            vec = (pos[i] - pos[j]) / d
            grad[i] += 0.5 * vec
            grad[j] += -0.5 * vec
        return -r_pair, -grad.flatten()
    else:
        s = np.sign(pos[wi, wk])
        if s == 0:
            s = 1.0
        grad[wi, wk] = -s
        return -r_wall, -grad.flatten()

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass using the hard-min objective."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            objective_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Better grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    # Evaluate initial
    val, _, _, _, _ = compute_min_dist(best_pos)
    best_r = val
    
    rng = np.random.default_rng(42)
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Decide start point with adaptive perturbation
        if attempts % 5 == 0:
            # Random start
            start_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        elif attempts % 5 == 1:
            # Small perturbation of best
            scale = 0.02
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 2:
            # Medium perturbation
            scale = 0.05
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 3:
            # Large perturbation
            scale = 0.1
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        else:
            # Very small perturbation for fine-tuning
            scale = 0.005
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        
        # Clip to stay inside bounds
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Try soft-min (global) and hard-min objectives
        # Use soft-min for most attempts to escape local minima, hard-min for polishing
        use_soft = (attempts % 3 != 0) # 2/3 soft, 1/3 hard
        
        try:
            if use_soft:
                # Adaptive beta? Start high (sharp) and lower if stuck?
                # For now, fixed beta=50 is a good balance for N=100
                res = minimize(
                    lambda x: soft_min_all_objective(x, beta=50.0)[0],
                    start_pos.flatten(),
                    jac=lambda x: soft_min_all_objective(x, beta=50.0)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            else:
                res = minimize(
                    objective_and_grad,
                    start_pos.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            
            if res.success or res.fun < -best_r:
                new_pos = res.x.reshape(N, 3)
                if np.all(np.isfinite(new_pos)):
                    val, _, _, _, _ = compute_min_dist(new_pos)
                    if val > best_r:
                        best_r = val
                        best_pos = new_pos
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=300, ftol=1e-12, gtol=1e-10)
        val, _, _, _, _ = compute_min_dist(polished_pos)
        if val > best_r:
            best_r = val
            best_pos = polished_pos
    
    return best_pos

# EVOLVE-BLOCK-END
