import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Computes the score for a given set of centers.
    Score = min(min_pair_dist / 2, min_wall_dist)
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin(v, beta):
    """
    Computes the soft-min of vector v with parameter beta.
    """
    min_v = np.min(v)
    exp_terms = np.exp(-beta * (v - min_v))
    log_sum = np.log(np.sum(exp_terms))
    return min_v - (1.0 / beta) * log_sum

def _objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes -softmin(gaps, beta).
    Returns (objective, gradient).
    """
    centers = x.reshape(N, 3)
    
    # 1. Pairwise gaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_gaps = dists / 2.0
    
    # 2. Wall gaps
    wall_gaps = 0.5 - np.abs(centers)
    
    # Combine all gaps into one vector
    # Indices: 0..1224 are pairs (i<j), 1225..1374 are walls (i,k)
    n_pairs = N * (N - 1) // 2
    n_walls = N * 3
    
    v_pairs = pair_gaps[np.triu_indices(N, k=1)]
    v_walls = wall_gaps.flatten()
    
    v = np.concatenate([v_pairs, v_walls])
    
    # Softmin calculation
    min_v = np.min(v)
    # Subtract min for numerical stability
    exp_terms = np.exp(-beta * (v - min_v))
    sum_exp = np.sum(exp_terms)
    # Probabilities p_k
    p = exp_terms / sum_exp
    
    # Objective
    log_sum = np.log(sum_exp)
    soft_min_val = min_v - (1.0 / beta) * log_sum
    obj = -soft_min_val
    
    # Gradient calculation
    # grad_obj = - sum_k p_k * grad_v_k
    # We accumulate grad_v_k contributions into grad_x
    grad_x = np.zeros_like(x)
    
    # Pair contributions
    # p_pairs = p[:n_pairs]
    p_pairs = p[:n_pairs]
    
    # We need to map flat pair indices back to (i, j)
    # This is expensive to do every iteration if done naively.
    # However, N=50 is small.
    # Alternative: Vectorized accumulation.
    
    # Let's use a loop for clarity and correctness first, then optimize if needed.
    # Actually, we can vectorize the accumulation using np.add.at or similar.
    # But constructing the index arrays for np.add.at every time is also costly.
    # Given N=50, 1225 pairs. A Python loop of 1225 iterations is ~1ms.
    # L-BFGS-B might call this 100-1000 times. 1-100ms per call is acceptable?
    # 1000 calls * 1ms = 1s. That's fine.
    # But we can do better.
    
    # Let's precompute the indices for pairs.
    # But we can't easily precompute in the function without passing them.
    # Let's just use the loop, it's robust.
    
    # Actually, let's try to vectorize the pair gradient accumulation.
    # diff is (N, N, 3). dists is (N, N).
    # unit_vectors = diff / dists[..., np.newaxis]
    # This has 0/0 on diagonal. We set dists diag to inf, so unit_vectors diag is 0.
    unit_vectors = diff / dists[..., np.newaxis]
    
    # We want to sum p_ij * unit_vectors[i,j] to grad[i]
    # and p_ij * unit_vectors[j,i] to grad[j] (which is -p_ij * unit_vectors[i,j])
    
    # p_matrix: (N, N) with p_pairs in upper triangle.
    p_matrix = np.zeros((N, N))
    # Fill upper triangle
    iu = np.triu_indices(N, k=1)
    p_matrix[iu] = p_pairs
    
    # grad_from_pairs[i] = sum_j p_ij * unit_vectors[i,j]
    # grad_from_pairs[j] = sum_i p_ij * unit_vectors[j,i] = - sum_i p_ij * unit_vectors[i,j]
    
    # So grad_from_pairs = p_matrix @ unit_vectors - unit_vectors.T @ p_matrix? No.
    # grad_from_pairs[i] = sum_j p_matrix[i,j] * unit_vectors[i,j]
    # This is element-wise multiplication then sum over j.
    # weighted_uv = p_matrix * unit_vectors
    # grad_from_pairs = np.sum(weighted_uv, axis=1)
    # Wait, unit_vectors[i,j] is vector (c_i - c_j)/d_ij.
    # So grad_from_pairs[i] = sum_j p_ij * (c_i - c_j)/d_ij.
    # This is correct for the contribution to c_i.
    
    weighted_uv = p_matrix * unit_vectors
    grad_pairs = np.sum(weighted_uv, axis=1)
    
    # Wall contributions
    p_walls = p[n_pairs:]
    # p_walls is flattened (N, 3)
    p_walls_matrix = p_walls.reshape(N, 3)
    
    # grad_wall[i,k] = p_ik * d(0.5 - |c_ik|)/dc_ik = p_ik * (-sign(c_ik))
    signs = np.sign(centers)
    # Handle 0
    signs[centers == 0] = 0.0
    
    grad_walls = -p_walls_matrix * signs
    
    # Total gradient of softmin
    grad_softmin = grad_pairs + grad_walls
    
    # Gradient of objective (which is -softmin)
    grad_obj = -grad_softmin.flatten()
    
    return obj, grad_obj

def _generate_bcc_lattice():
    """
    Generate a BCC lattice initialization.
    BCC has 2 points per unit cell.
    We want ~50 points.
    A 3x3x3 grid of BCC cells has 27 * 2 = 54 points.
    We can select 50 of them.
    """
    m = 3
    # Corner points
    corners = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                corners.append((i/m - 0.5, j/m - 0.5, k/m - 0.5))
    
    # Body center points
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                centers.append((i/m - 0.5 + 0.5/m, j/m - 0.5 + 0.5/m, k/m - 0.5 + 0.5/m))
    
    all_points = np.array(corners + centers)
    
    # Randomly select N points
    # It's better to keep the structure, so maybe just take the first N?
    # Or random shuffle.
    idx = np.random.permutation(len(all_points))
    selected = all_points[idx[:N]]
    
    # Add small noise to break symmetry
    selected += np.random.normal(0, 0.01, selected.shape)
    selected = np.clip(selected, -0.49, 0.49)
    
    return selected.flatten()

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_centers = None
    best_score = -1.0
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Define restarts
    for i in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial guess
        if i % 3 == 0:
            # BCC Lattice
            x0 = _generate_bcc_lattice()
        elif i % 3 == 1:
            # Random uniform
            x0 = np.random.uniform(-0.5, 0.5, (N, 3)).flatten()
        else:
            # Perturbed Grid (4x4x4)
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            grid = grid[idx[:N]]
            grid += np.random.normal(0, 0.02, grid.shape)
            grid = np.clip(grid, -0.49, 0.49)
            x0 = grid.flatten()
            
        # Optimization Stages
        # We use a sequence of beta values (annealing)
        betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
        
        current_x = x0
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                # Use L-BFGS-B with analytical gradient
                res = minimize(_objective_and_grad, current_x, args=(beta,), 
                               method='L-BFGS-B', jac=True,
                               bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                current_x = res.x
            except Exception:
                break
        
        if time.time() - start_time > time_limit:
            break
            
        # Evaluate Result
        centers = current_x.reshape(N, 3)
        
        # Check validity (finite values)
        if not np.all(np.isfinite(centers)):
            continue
            
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
        # Optional: Early termination if we reach a very high score
        if best_score > 0.136:
             # We can try to polish further or just break?
             # Let's keep searching to see if we can beat it, but maybe reduce restarts?
             # For now, just keep going until time limit.
             pass

    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
