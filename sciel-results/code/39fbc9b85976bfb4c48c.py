"""Improved packing: grid init + L-BFGS-B on smooth soft-min objective with deadline guard."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(c):
    """Compute the radius for a given set of centres (must be finite, inside cube)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(np.abs(c) > 0.5):
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _objective_and_grad(c_flat, beta):
    """
    Compute objective value and gradient for the soft-min surrogate.
    Objective: minimize -softmin(terms) where terms = pairwise dists/2 and wall dists.
    Returns (fval, grad) where grad has shape (N*3,).
    """
    c = c_flat.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dists = np.sqrt(np.maximum(dists_sq, 1e-20))
    np.fill_diagonal(dists, np.inf)
    
    pair_terms = dists / 2.0  # (N, N)
    wall_terms = 0.5 - np.abs(c)  # (N, 3)
    
    # Flatten and collect all terms
    pair_flat = pair_terms.ravel()
    pair_mask = np.isfinite(pair_flat)
    pair_flat_valid = pair_flat[pair_mask]
    
    wall_flat = wall_terms.ravel()
    
    all_terms = np.concatenate([pair_flat_valid, wall_flat])
    if len(all_terms) == 0:
        return 0.0, np.zeros_like(c_flat)
    
    # Numerically stable soft-min computation
    m = np.min(all_terms)
    exp_terms = np.exp(-(all_terms - m) / beta)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    softmin_val = m - beta * log_sum
    
    # Objective is -softmin
    fval = -softmin_val
    
    # Gradient: d(softmin)/d(x_k) = exp(-x_k/beta) / sum(exp(-x_j/beta))
    # d(objective)/d(x_k) = -d(softmin)/d(x_k) = -exp(-x_k/beta) / sum_exp
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(c_flat)
    
    # Back-propagate through pairwise terms
    # pair_flat_valid corresponds to off-diagonal elements of pair_terms
    # We need to map each weight back to its (i,j) pair
    n_pairs = len(pair_flat_valid)
    pair_weights = weights[:n_pairs]
    wall_weights = weights[n_pairs:]
    
    # Reconstruct the full pair_weights matrix
    pair_w_matrix = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j and pair_mask[idx * N + j] if idx < n_pairs else False:
                pass
            idx += 1
    # This approach is too slow. Let's use a different strategy.
    # Instead, compute gradient directly using the structure.
    
    # Clear and recompute
    grad = np.zeros_like(c_flat)
    
    # For pairwise: term_{ij} = |c_i - c_j| / 2
    # d(term_{ij})/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(term_{ij})/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    # 
    # d(softmin)/d(c_i) = sum_{j != i} w_{ij} * (c_i - c_j) / (2 * dists[i,j])
    # where w_{ij} = exp(-term_{ij}/beta) / sum_all
    #
    # d(objective)/d(c_i) = -d(softmin)/d(c_i)
    
    # Compute weights matrix
    pair_w = np.zeros((N, N))
    pair_w[pair_mask.reshape(N, N)] = pair_weights
    
    # For each point i, accumulate contributions from all j
    for i in range(N):
        # Contributions from pairwise terms
        # d(softmin)/d(c_i) from pairs = sum_j w_{ij} * (c_i - c_j) / (2 * dists[i,j])
        contrib = np.zeros(3)
        for j in range(N):
            if i != j and dists[i, j] < np.inf:
                w = pair_w[i, j]
                if w > 0:
                    d = c[i] - c[j]
                    dist = dists[i, j]
                    contrib += w * d / (2.0 * dist)
        # Wall contribution: d(softmin)/d(c_i) from walls
        # wall_terms[i,k] = 0.5 - |c_i[k]|
        # d(wall_terms[i,k])/d(c_i[k]) = -sign(c_i[k])
        # d(softmin)/d(c_i[k]) = w_wall[i,k] * (-sign(c_i[k]))
        for k in range(3):
            w = wall_weights[i * 3 + k]
            if w > 0:
                contrib[k] += w * (-np.sign(c[i, k]))
        
        # d(objective)/d(c_i) = -contrib
        grad[i * 3:(i + 1) * 3] = -contrib
    
    return fval, grad


def _initial_grid():
    """Generate an initial grid of N points in the cube."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    c0 = _initial_grid().astype(float)
    best_c = c0.copy()
    best_score = _score(best_c)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Beta annealing schedule: start loose, tighten progressively
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 25.0:
            break
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                c0.reshape(-1),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 80000}
            )
            
            new_c = result.x.reshape(N, 3)
            new_score = _score(new_c)
            if new_score > best_score:
                best_c = new_c
                best_score = new_score
            
            c0 = best_c.copy()
            
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, 1000.0)[0],
                best_c.reshape(-1),
                jac=lambda x: _objective_and_grad(x, 1000.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 150000}
            )
            new_c = result.x.reshape(N, 3)
            new_score = _score(new_c)
            if new_score > best_score:
                best_c = new_c
                best_score = new_score
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_c)) or np.any(np.abs(best_c) > 0.5):
        return _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
