import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _compute_distances_and_weights(centers, beta):
    """
    Computes distances D, and weights w for soft-min.
    Returns:
    D: array of all constraint values (pairwise half-dists, boundary dists)
    w: array of weights
    """
    centers = np.asarray(centers).reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    # Use broadcasting to compute all pairwise differences
    # diff[i, j, k] = centers[i, k] - centers[j, k]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Only take upper triangle for i < j
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    # dist to each face: 0.5 - |c_ik|
    abs_centers = np.abs(centers)
    # shape (n, 3)
    bound_dists = CUBE_HALF - abs_centers
    
    # Flatten all distances into one vector
    D = np.concatenate([pair_dists, bound_dists.ravel()])
    
    # Compute weights w_k = exp(-beta * D_k) / sum(exp(-beta * D_j))
    # To avoid overflow/underflow, subtract max
    # Note: D can be 0 or negative if overlapping/outside, but we assume valid start
    # Actually, if D is small, -beta*D is large positive.
    # Let's use log-sum-exp trick.
    # We want w_k proportional to exp(-beta * D_k).
    # Let M = min(D). Then exp(-beta * D_k) = exp(-beta * (D_k - M)) * exp(-beta * M).
    # The term exp(-beta * M) cancels in normalization.
    # So we can just use exp(-beta * (D_k - M)).
    
    # However, if beta is large, D_k - M is small positive.
    # If D has large spread, some terms might underflow to 0, which is fine.
    # If D_k is very large, term is 0.
    
    # We need gradients, so we need to know which D corresponds to which center coordinate.
    # It is easier to compute the gradient directly in the objective function 
    # or return the weights and indices.
    
    # For L-BFGS-B, we need the gradient of the objective function f = -softmin(D).
    # f = (1/beta) * log(sum(exp(-beta * D_k)))
    # grad_f = sum_k w_k * grad_D_k
    # where w_k = exp(-beta * D_k) / sum(exp(-beta * D_j))
    
    # Let's return D, w, and the indices/structure to compute grad.
    # Or better, just compute the gradient in a separate function.
    
    return D, w, pair_dists, bound_dists, iu, diff, dists

def _softmin_objective_and_grad(centers_flat, beta):
    """
    Objective function and its gradient.
    centers_flat: shape (3*N,)
    """
    centers = centers_flat.reshape(-1, 3)
    n = len(centers)
    
    # 1. Compute pairwise distances and gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    pair_diffs = diff[iu] # shape (M, 3)
    
    # Gradient of pair_dist w.r.t centers
    # d/dc_i (||c_i - c_j||/2) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d/dc_j (||c_i - c_j||/2) = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # pair_dists shape (M,)
    # pair_diffs shape (M, 3)
    # grad_pair shape (M, 3)
    grad_pair = pair_diffs / (2.0 * dists[iu][:, np.newaxis])
    
    # 2. Compute boundary distances and gradients
    abs_centers = np.abs(centers)
    bound_dists = CUBE_HALF - abs_centers # shape (N, 3)
    
    # Gradient of (0.5 - |c_ik|) w.r.t c_ik is -sign(c_ik)
    # If c_ik is 0, sign is 0, but derivative is undefined. 
    # Usually we treat it as 0 or use a small epsilon.
    # Since we start from grid (not on boundary exactly, but centers are 0.5/m - 0.5),
    # they are not 0.
    sign_centers = np.sign(centers)
    grad_bound = -sign_centers # shape (N, 3)
    
    # 3. Combine all D values
    D = np.concatenate([pair_dists, bound_dists.ravel()])
    
    # 4. Compute weights w
    # w_k = exp(-beta * D_k) / sum(exp(-beta * D_j))
    # Use log-sum-exp for stability
    # log_sum_exp(-beta * D)
    # max_val = np.max(-beta * D)
    # log_sum = max_val + np.log(np.sum(np.exp(-beta * D - max_val)))
    # log_w = -beta * D - log_sum
    # w = np.exp(log_w)
    
    # Simpler:
    exp_terms = np.exp(-beta * D)
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        # Fallback: uniform
        w = np.ones_like(D) / len(D)
    else:
        w = exp_terms / sum_exp
    
    # 5. Compute gradient of objective f = -softmin(D) = (1/beta) log(sum exp(-beta D))
    # grad_f = sum_k w_k * grad_D_k
    # Note: f = (1/beta) * log(sum(exp(-beta D)))
    # d f / d D_k = (1/beta) * (1/sum) * (-beta) * exp(-beta D_k) = - w_k
    # Wait.
    # S = -1/beta log(sum exp(-beta D))
    # dS/dD_k = -1/beta * 1/sum * (-beta) exp(-beta D_k) = w_k
    # We want to maximize S, so minimize f = -S.
    # df/dD_k = -w_k
    # df/dX = sum_k (-w_k) * dD_k/dX
    
    grad_flat = np.zeros_like(centers_flat)
    
    # Add contributions from pairs
    # For each pair (i, j) in iu:
    # dD_ij/dc_i = grad_pair[k]
    # dD_ij/dc_j = -grad_pair[k]
    # Contribution to grad_f:
    # -w[k] * grad_pair[k] to c_i
    # -w[k] * (-grad_pair[k]) = w[k] * grad_pair[k] to c_j
    
    # We need to scatter these gradients back to the full array.
    # iu[0] are i indices, iu[1] are j indices.
    idx_i = iu[0]
    idx_j = iu[1]
    
    # Use np.add.at for accumulation
    # grad_pair shape (M, 3)
    # w_pair = w[:M]
    w_pair = w[:len(pair_dists)]
    
    # grad_contrib_i = -w_pair[:, np.newaxis] * grad_pair
    # grad_contrib_j = w_pair[:, np.newaxis] * grad_pair
    
    np.add.at(grad_flat, (idx_i * 3, 0), -w_pair * grad_pair[:, 0])
    np.add.at(grad_flat, (idx_i * 3 + 1, 0), -w_pair * grad_pair[:, 1])
    np.add.at(grad_flat, (idx_i * 3 + 2, 0), -w_pair * grad_pair[:, 2])
    
    np.add.at(grad_flat, (idx_j * 3, 0), w_pair * grad_pair[:, 0])
    np.add.at(grad_flat, (idx_j * 3 + 1, 0), w_pair * grad_pair[:, 1])
    np.add.at(grad_flat, (idx_j * 3 + 2, 0), w_pair * grad_pair[:, 2])
    
    # Add contributions from boundaries
    # D_bound shape (N, 3)
    # grad_bound shape (N, 3)
    # w_bound = w[M:] reshaped to (N, 3)
    w_bound = w[len(pair_dists):].reshape(n, 3)
    
    # d f / d c_ik = -w_bound[i, k] * grad_bound[i, k]
    grad_bound_contrib = -w_bound * grad_bound
    
    grad_flat = grad_flat.reshape(n, 3) + grad_bound_contrib
    grad_flat = grad_flat.ravel()
    
    # Objective value
    # f = (1/beta) * log(sum_exp)
    # If sum_exp is 0, return a large penalty? 
    # But we assume valid inputs.
    val = (1.0 / beta) * np.log(sum_exp) if sum_exp > 0 else 1000.0
    
    return val, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_grid = grid[:N].copy()
    
    best_centers = initial_grid
    best_radius = _calc_radius(initial_grid)
    
    # Random seed for reproducibility? No, let it vary or fix? 
    # Fixed is safer for evaluation consistency, but random restarts need different seeds.
    # We will use a fixed seed for the first few, then random.
    
    rng = np.random.default_rng(42)
    
    # Beta schedule: start low, increase.
    # Low beta = smooth, high beta = sharp (approximates min).
    betas = [10.0, 50.0, 100.0, 200.0, 400.0]
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the grid
        # Noise scale should be small enough to keep spheres valid (non-overlapping)
        # Grid spacing is ~ 1/5 = 0.2. Radius ~ 0.1.
        # Perturbation ~ 0.01 is safe.
        noise_scale = 0.01
        centers = initial_grid + rng.normal(0, noise_scale, size=initial_grid.shape)
        
        # Ensure centers are within bounds
        centers = np.clip(centers, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Optimize
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Flatten for scipy
            x0 = centers.ravel()
            bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
            
            try:
                res = minimize(
                    _softmin_objective_and_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
                )
                centers = res.x.reshape(-1, 3)
            except Exception:
                pass
            
            # Update best if improved
            current_radius = _calc_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                
    # Final polish: try to push the best centers further with a very high beta
    # This helps refine the minimum distance
    if time.time() < deadline - 2.0:
        centers = best_centers.copy()
        x0 = centers.ravel()
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        try:
            res = minimize(
                _softmin_objective_and_grad,
                x0,
                args=(1000.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            centers = res.x.reshape(-1, 3)
            current_radius = _calc_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        except Exception:
            pass
            
    return best_centers

def _calc_radius(centers):
    """Calculate the actual radius (min distance) for scoring."""
    centers = np.asarray(centers).reshape(-1, 3)
    n = len(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    
    # Boundary
    abs_centers = np.abs(centers)
    min_bound = np.min(CUBE_HALF - abs_centers)
    
    return min(min_pair, min_bound)
