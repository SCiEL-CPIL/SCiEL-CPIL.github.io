"""Packing 200 equal circles in a unit square using smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200


def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    wall = 0.5 - np.abs(centers).max(axis=1)
    if np.any(wall < 0):
        return 0.0
    # Pairwise distances
    # Use a chunked approach to avoid memory issues for large N
    min_pair_dist = np.inf
    for i in range(N):
        diffs = centers[i] - centers[i+1:]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if dists.size > 0:
            min_pair_dist = min(min_pair_dist, dists.min())
    r = min(wall.min(), min_pair_dist / 2.0)
    return r


def _soft_min(values, beta):
    """Smooth approximation of the minimum using log-sum-exp."""
    # log-sum-exp trick for numerical stability
    m = np.min(values)
    if m == -np.inf:
        return -np.inf
    shifted = values - m
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    return m - np.log(np.sum(np.exp(beta * shifted))) / beta


def _objective_and_grad(params, beta=50.0):
    """
    Objective: minimize -r where r is approximated via soft-min of all constraints.
    params: (N*2,) array of centers
    Returns: (objective_value, gradient)
    """
    centers = params.reshape(N, 2)
    
    # Wall distances
    # For each point and each wall: 0.5 - |x|, 0.5 - |y|
    # We want the minimum of all these
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    
    # Pairwise distances / 2
    # This is expensive; compute all pairs
    # For N=200, that's ~20000 pairs, manageable
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Combine all constraint values
    # We want soft-min of: all wall_x, all wall_y, all pair_vals (upper triangle)
    # To keep it simple, let's just use a combined vector
    # Actually, for efficiency, let's compute the soft-min more carefully
    
    # Wall values
    wall_vals = np.concatenate([wall_x, wall_y])
    
    # Pair values (only upper triangle to avoid double counting, but for soft-min it doesn't matter much)
    # Let's use all off-diagonal elements
    pair_flat = pair_vals[np.triu_indices(N, k=1)]
    
    all_vals = np.concatenate([wall_vals, pair_flat])
    
    # Soft-min
    r_soft = _soft_min(all_vals, beta)
    
    # Objective is -r_soft (we maximize r)
    obj = -r_soft
    
    # Gradient: d(-r_soft)/d(params)
    # r_soft = m - (1/beta) * log(sum(exp(beta * (v_i - m))))
    # where m = min(all_vals)
    # dr_soft/dv_i = (1/beta) * exp(beta*(v_i - m)) / sum(exp(beta*(v_j - m)))
    #               = softmax(beta * (all_vals - m))_i / beta... wait
    # Let's recompute:
    # f(v) = -1/beta * log(sum_j exp(beta * (v_j - m))) + m
    # But m is also a function of v. This is getting complex.
    # 
    # Alternative: use the standard soft-min formula:
    # softmin(v, beta) = -1/beta * log(sum_j exp(-beta * v_j))
    # d(softmin)/dv_i = exp(-beta * v_i) / sum_j exp(-beta * v_j)
    # 
    # Let's use this form instead for cleaner gradients.
    
    # Recompute with the other form
    neg_beta_vals = -beta * all_vals
    # For numerical stability
    max_neg = np.max(neg_beta_vals)
    exp_neg = np.exp(neg_beta_vals - max_neg)
    sum_exp = np.sum(exp_neg)
    
    # softmin = -1/beta * (log(sum_exp) + max_neg)
    # = -1/beta * log(sum_exp) - max_neg/beta
    
    # Gradient wrt each v_i:
    # d(softmin)/dv_i = exp(-beta * v_i) / sum_j exp(-beta * v_j)
    # = exp_neg_i / sum_exp
    
    weights = exp_neg / sum_exp
    
    # Now we need to backpropagate through all_vals to params
    # all_vals = [wall_x, wall_y, pair_flat]
    grad_wall_x = np.zeros(N)
    grad_wall_y = np.zeros(N)
    grad_pair = np.zeros_like(pair_flat)
    
    # wall_x[i] = 0.5 - |centers[i,0]|
    # d(wall_x[i])/d(centers[i,0]) = -sign(centers[i,0])
    grad_wall_x_vals = weights[:N]  # corresponds to wall_x
    grad_wall_y_vals = weights[N:2*N]  # corresponds to wall_y
    grad_pair_vals = weights[2*N:]  # corresponds to pair_flat
    
    # Build gradient for centers
    grad_centers = np.zeros((N, 2))
    
    # From wall_x
    signs_x = np.sign(centers[:, 0])
    signs_x[signs_x == 0] = 1.0
    grad_centers[:, 0] += -grad_wall_x_vals * signs_x
    
    # From wall_y
    signs_y = np.sign(centers[:, 1])
    signs_y[signs_y == 0] = 1.0
    grad_centers[:, 1] += -grad_wall_y_vals * signs_y
    
    # From pair distances
    # pair_flat contains dists[i,j]/2 for i<j
    # d(dist(i,j)/2)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dist(i,j))
    # d(dist(i,j)/2)/d(centers[j]) = (centers[j] - centers[i]) / (2 * dist(i,j))
    
    # We need to map back from pair_flat to (i,j) pairs
    # This is expensive but necessary
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    # For each pair (i,j), the contribution to grad
    # pair_flat[k] corresponds to (idx_i[k], idx_j[k])
    # dist = dists[idx_i[k], idx_j[k]]
    # grad contribution to centers[idx_i[k]]: (centers[idx_i[k]] - centers[idx_j[k]]) / (2 * dist) * grad_pair_vals[k]
    # grad contribution to centers[idx_j[k]]: (centers[idx_j[k]] - centers[idx_i[k]]) / (2 * dist) * grad_pair_vals[k]
    
    # Vectorized approach:
    pair_dists = dists[idx_i, idx_j]
    # Avoid division by zero
    pair_dists_safe = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    
    diff_vec = centers[idx_i] - centers[idx_j]  # shape (n_pairs, 2)
    factor = grad_pair_vals / (2.0 * pair_dists_safe)  # shape (n_pairs,)
    
    # Contribution to centers[idx_i]: factor * diff_vec
    # Contribution to centers[idx_j]: -factor * diff_vec
    
    # Use np.add.at for accumulation
    np.add.at(grad_centers, idx_i, factor[:, np.newaxis] * diff_vec)
    np.add.at(grad_centers, idx_j, -factor[:, np.newaxis] * diff_vec)
    
    # Total gradient for params (flattened centers)
    # Objective is -softmin, so gradient is -d(softmin)/d(params)
    grad_params = -grad_centers.flatten()
    
    return obj, grad_params


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial guess: hexagonal packing
    # For 200 circles, roughly sqrt(200) ≈ 14.14, so about 14x14 grid
    # Hexagonal: rows offset
    m = int(np.ceil(np.sqrt(N)))
    
    # Create hexagonal lattice
    centers = []
    row = 0
    while len(centers) < N:
        x_offset = 0.0 if row % 2 == 0 else 0.5
        n_in_row = m
        for col in range(n_in_row):
            if len(centers) >= N:
                break
            x = (col + x_offset + 0.5) / (m + 0.5) - 0.5
            y = (row + 0.5) / (m + 0.5) - 0.5
            centers.append([x, y])
        row += 1
    
    centers = np.array(centers[:N])
    
    # Initial radius
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Try multiple beta values and restarts
    betas = [20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        
        # Multiple restarts with different initial conditions
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb initial positions slightly for different restarts
            if restart > 0:
                rng = np.random.RandomState(restart * 42 + 7)
                perturbed = best_centers + rng.normal(0, 0.01, size=best_centers.shape)
                # Keep within bounds
                perturbed = np.clip(perturbed, -0.49, 0.49)
            else:
                perturbed = centers.copy()
            
            params = perturbed.flatten()
            
            try:
                result = minimize(
                    _objective_and_grad,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:
                    new_centers = result.x.reshape(N, 2)
                    new_radius = _compute_radius(new_centers)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                        
            except Exception as e:
                pass
        
        # After each beta, try to increase radius via binary search
        if time.time() > deadline - 5.0:
            break
        
        # Try to push radius higher
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Current best radius
            current_r = _compute_radius(best_centers)
            
            # Try to optimize with a slightly higher target
            # This is a heuristic: just re-optimize from current best
            params = best_centers.flatten()
            try:
                result = minimize(
                    _objective_and_grad,
                    params,
                    args=(100.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success:
                    new_centers = result.x.reshape(N, 2)
                    new_radius = _compute_radius(new_centers)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
                        
            except Exception as e:
                pass
    
    # Final validation and cleanup
    # Ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
