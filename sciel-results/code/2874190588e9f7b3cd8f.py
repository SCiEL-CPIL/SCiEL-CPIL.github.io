import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(p):
    """
    Compute the maximum radius for given positions p.
    Returns -1 if invalid.
    """
    if not np.all(np.isfinite(p)):
        return -1.0
    if np.any(p < -0.5 - 1e-9) or np.any(p > 0.5 + 1e-9):
        return -1.0
    
    # Wall distance
    wall_dists = 0.5 - np.abs(p)
    min_wall = np.min(wall_dists)
    
    # Pairwise distance
    # Vectorized pairwise distance calculation
    # p is (N, 3)
    # diff is (N, N, 3)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _soft_min_objective_and_grad(p, beta):
    """
    Objective: Minimize the soft-min of pairwise distances and wall distances.
    This approximates maximizing the minimum distance.
    """
    p = p.reshape(N, 3)
    
    # 1. Pairwise distances
    # Compute all pairwise differences
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Create a mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    # Exponential terms for pairs
    # To avoid underflow/overflow issues with large beta, we can shift, 
    # but for beta up to ~1000 and dist ~0.1, exp(-100) is tiny.
    # The sum will be dominated by the smallest distances.
    
    # 2. Wall distances
    # For each point, distance to nearest wall
    # wall_dist_i = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # Actually, min_k (0.5 - |c_ik|)
    abs_p = np.abs(p)
    wall_dists = 0.5 - abs_p  # shape (N, 3)
    # We need the min over k for each i? 
    # The constraint is r <= 0.5 - |c_ik| for all k.
    # So the limiting wall distance for point i is min_k (0.5 - |c_ik|).
    # The global min is min_i (min_k (0.5 - |c_ik|)).
    # In the log-sum-exp, we sum over all individual wall constraints.
    # Sum_i Sum_k exp(-beta * (0.5 - |c_ik|))
    
    exp_pairs = np.exp(-beta * pair_dists)
    exp_walls = np.exp(-beta * wall_dists)
    
    sum_pairs = np.sum(exp_pairs)
    sum_walls = np.sum(exp_walls)
    
    total_sum = sum_pairs + sum_walls
    
    # Avoid log(0)
    if total_sum == 0:
        return 0.0, np.zeros_like(p)
    
    obj = (1.0 / beta) * np.log(total_sum)
    
    # Gradient
    # d/dp log(sum(exp(-beta x))) = - (sum(exp(-beta x) * grad(x))) / sum(exp(-beta x))
    
    grad = np.zeros_like(p)
    
    # Gradient contribution from pairs
    # x_ij = ||p_i - p_j||
    # grad_{p_i} x_ij = (p_i - p_j) / x_ij
    # grad_{p_j} x_ij = (p_j - p_i) / x_ij
    
    # We need to accumulate weights * grad for each pair.
    # We have pair_dists and exp_pairs.
    # We need the vectors.
    
    # To be efficient, we can iterate or use advanced indexing.
    # Given N=50, looping over pairs for gradient accumulation is acceptable 
    # if we pre-compute the normalized vectors.
    
    # Let's get the indices
    i_idx, j_idx = np.where(mask)
    
    # diff[i, j] is p_i - p_j
    vecs = diff[i_idx, j_idx] # shape (M, 3)
    dists_val = pair_dists # shape (M,)
    
    # Avoid division by zero
    safe_dists = np.where(dists_val < 1e-8, 1e-8, dists_val)
    unit_vecs = vecs / safe_dists[:, np.newaxis]
    
    # Weights for gradient
    # w_ij = exp_pairs[i] / total_sum
    # Note: The derivative of log(sum) is sum(w_k * grad_k)
    # Here the sum is over all terms.
    # The term for pair (i,j) is exp(-beta * d_ij).
    # Its gradient w.r.t p_i is -beta * exp(-beta * d_ij) * (p_i - p_j)/d_ij
    # Its gradient w.r.t p_j is -beta * exp(-beta * d_ij) * (p_j - p_i)/d_ij
    
    # The total gradient of the objective (1/beta * log S) is:
    # (1/beta) * (1/S) * sum_terms (d term / dp)
    # d term / dp_i = -beta * exp(-beta d) * unit_vec
    # So contribution to grad_i is (1/beta) * (1/S) * (-beta * exp * unit_vec) = - (exp/S) * unit_vec
    
    weights = exp_pairs / total_sum
    
    # Accumulate gradients for i
    # grad_i += sum(weights * unit_vec)
    # grad_j -= sum(weights * unit_vec)
    
    np.add.at(grad, i_idx, weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -weights[:, np.newaxis] * unit_vecs)
    
    # Gradient contribution from walls
    # Term: exp(-beta * (0.5 - |p_ik|))
    # Let x = 0.5 - |p_ik|
    # d x / dp_ik = -sign(p_ik)
    # d term / dp_ik = -beta * exp(-beta x) * (-sign(p_ik)) = beta * exp(-beta x) * sign(p_ik)
    # Contribution to grad: (1/beta) * (1/S) * (beta * exp * sign) = (exp/S) * sign
    
    wall_weights = exp_walls / total_sum
    signs = np.sign(p)
    
    # grad_ik += wall_weights[i,k] * sign(p_ik)
    grad += wall_weights * signs
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    best_r = -1.0
    best_pos = None
    
    # Initializations to try
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N].copy()
    
    # 2. Random
    rng = np.random.RandomState(42)
    init_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    # 3. BCC-like
    # Generate a BCC lattice and take the first N points
    # BCC points: corners and body centers
    # Scale to fit in cube
    init_bcc = None
    # Simple BCC generation
    bcc_points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                bcc_points.append([i/3.0 - 0.5, j/3.0 - 0.5, k/3.0 - 0.5])
                # Body center
                bcc_points.append([i/3.0 - 0.5 + 1/6.0, j/3.0 - 0.5 + 1/6.0, k/3.0 - 0.5 + 1/6.0])
    bcc_points = np.array(bcc_points)
    # Shuffle and take N
    rng.shuffle(bcc_points)
    init_bcc = bcc_points[:N].copy()
    
    initializations = [init_grid, init_rand, init_bcc]
    
    # Beta schedule
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        current_pos = init_pos.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    _soft_min_objective_and_grad,
                    current_pos,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_pos = res.x
            except Exception as e:
                # If optimization fails, keep current pos
                pass
                
        # Evaluate the final position
        r = _compute_radius(current_pos)
        if r > best_r:
            best_r = r
            best_pos = current_pos.copy()
            
        # If time permits, do a few random restarts from the best position found so far
        # This helps escape local minima
        time_left = deadline - time.time()
        if time_left > 20:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                # Perturb the best position
                perturbed = best_pos + rng.randn(N, 3) * 0.02
                # Clip to bounds
                perturbed = np.clip(perturbed, -0.49, 0.49)
                
                # Run a short optimization with high beta
                try:
                    res = minimize(
                        _soft_min_objective_and_grad,
                        perturbed,
                        args=(2000,),
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    new_pos = res.x
                    new_r = _compute_radius(new_pos)
                    if new_r > best_r:
                        best_r = new_r
                        best_pos = new_pos.copy()
                except Exception:
                    pass

    # Final fallback
    if best_pos is None or best_r < 0:
        best_pos = init_grid
        best_r = _compute_radius(best_pos)
        
    return best_pos
