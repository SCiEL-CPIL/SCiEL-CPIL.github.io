import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def soft_min_objective(x, k):
    """
    Objective function for scipy.optimize.minimize.
    x is a flattened array of shape (N*2,).
    Minimizes -soft_min(distances).
    """
    points = x.reshape(N, 2)
    
    # Calculate distances
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero in gradient if needed, but for value sqrt is fine
    # We need all pairwise distances for the soft min
    # Only upper triangle to avoid double counting? 
    # The soft min sum is over all constraints. 
    # If we include both (i,j) and (j,i), the weight is doubled, which is fine for the gradient direction, 
    # but we must be consistent. 
    # Let's just use all i < j.
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
    
    # Boundary
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    # To prevent overflow/underflow in exp, shift by min
    min_d = np.min(all_dists)
    exp_vals = np.exp(-k * (all_dists - min_d))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(x)
        
    soft_min_val = min_d - (1.0 / k) * np.log(sum_exp)
    obj_val = -soft_min_val
    
    # Gradient
    # d(obj)/dx = - d(soft_min)/dx
    # d(soft_min)/d(dist_m) = - exp_m / sum_exp
    # So d(obj)/d(dist_m) = exp_m / sum_exp
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # Pairwise gradients
    # For each pair (i, j) with weight w:
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
    # contrib to grad_i += w * (x_i - x_j) / dist_ij
    # contrib to grad_j += w * (x_j - x_i) / dist_ij
    
    # Vectorized update
    # We have weights for pairs.
    # dists_sq is full matrix. We extracted pair_dists.
    # We need the vectors.
    # diff[i_idx, j_idx] gives x_i - x_j.
    v_ij = diff[i_idx, j_idx]
    # dists are pair_dists
    # Avoid division by zero
    mask = pair_dists > 1e-9
    if np.any(mask):
        v_norm = v_ij[mask] / pair_dists[mask, np.newaxis]
        w = weights[:len(i_idx)][mask, np.newaxis]
        
        # Accumulate into grad
        # grad is flattened. 
        # It's easier to accumulate into a (N,2) array and flatten at the end.
        grad_matrix = np.zeros((N, 2))
        
        # Use add.at for efficiency
        np.add.at(grad_matrix, i_idx[mask], v_norm * w)
        np.add.at(grad_matrix, j_idx[mask], -v_norm * w)
        
        grad[:N*2] = grad_matrix.flatten()
        
    # Boundary gradients
    # d(b_i)/dx_i = -x_i / |x_i|
    # contrib to grad_i += w_bound_i * (-x_i / |x_i|)
    bound_weights = weights[len(i_idx):]
    rad_safe = np.where(rad > 1e-9, rad, 1.0)
    bound_grad_matrix = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
    
    grad_matrix = np.zeros((N, 2))
    np.add.at(grad_matrix, np.arange(N), bound_grad_matrix)
    
    # Wait, I re-initialized grad_matrix. I need to combine them.
    # Let's refactor the gradient calculation to be cleaner.
    
    return obj_val, grad

def soft_min_objective_and_grad(x, k):
    points = x.reshape(N, 2)
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
    
    # Boundary
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    min_d = np.min(all_dists)
    # Shift to avoid underflow
    exp_vals = np.exp(-k * (all_dists - min_d))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(x)
        
    soft_min_val = min_d - (1.0 / k) * np.log(sum_exp)
    obj_val = -soft_min_val
    
    weights = exp_vals / sum_exp
    
    grad_matrix = np.zeros((N, 2))
    
    # Pairwise
    mask = pair_dists > 1e-9
    if np.any(mask):
        v_ij = diff[i_idx[mask], j_idx[mask]]
        w = weights[:len(i_idx)][mask, np.newaxis]
        v_norm = v_ij / pair_dists[mask, np.newaxis]
        
        np.add.at(grad_matrix, i_idx[mask], v_norm * w)
        np.add.at(grad_matrix, j_idx[mask], -v_norm * w)
        
    # Boundary
    bound_weights = weights[len(i_idx):]
    rad_safe = np.where(rad > 1e-9, rad, 1.0)
    bound_grad = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
    grad_matrix += bound_grad
    
    return obj_val, grad_matrix.flatten()

def polish(points, time_budget):
    start_time = time.time()
    x0 = points.flatten()
    
    # Anneal k
    k_values = [10, 50, 100, 200, 500]
    
    current_x = x0
    for k in k_values:
        if time.time() - start_time > time_budget:
            break
            
        res = minimize(
            soft_min_objective_and_grad, 
            current_x, 
            args=(k,), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        current_x = res.x
        
    # Final projection to ensure validity
    final_points = current_x.reshape(N, 2)
    rad = np.linalg.norm(final_points, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        final_points[mask] /= rad[mask, np.newaxis]
        
    return final_points, compute_score(final_points)

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # 1. Initialization
    # Hexagonal lattice
    # Approximate radius based on area
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
    
    # 2. Optimization
    # We can run a few restarts if time permits, but L-BFGS-B is quite strong.
    # Let's do 1 main run, and if time allows, a perturbed run.
    
    # Phase 1: Coarse Polish
    time_for_polish = time_limit - (time.time() - start_time) - 5.0
    if time_for_polish > 0:
        polished_points, polished_score = polish(best_points, time_for_polish)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # Phase 2: If time remains, try a random perturbation and polish again
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        perturbed = best_points + np.random.normal(0, 0.01, size=best_points.shape)
        rad = np.linalg.norm(perturbed, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            perturbed[mask] /= rad[mask, np.newaxis]
            
        polished2, score2 = polish(perturbed, remaining - 2.0)
        if score2 > best_score:
            best_score = score2
            best_points = polished2
            
    return best_points
