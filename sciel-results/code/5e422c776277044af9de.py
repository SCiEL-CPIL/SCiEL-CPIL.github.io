"""
Structural change: Replace soft-min annealing with a two-phase approach:
1. Initialize with hexagonal lattice.
2. Phase 1: Soft-min annealing (coarser, fewer steps) to escape bad local minima.
3. Phase 2: Direct optimization of the hard min via a smooth approximation
   that focuses only on the critical (tightest) constraints, using a
   "log-sum-exp" on a small subset of near-critical pairs.
4. Final polish: Iterative local search on the tightest constraints only.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice clipped to the unit circle."""
    points = []
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _critical_subset(p, k=50):
    """
    Return indices of the k most critical constraints.
    Constraints: pair distances (i<j) and boundary distances.
    Returns: (pair_indices, boundary_indices, pair_dists, bound_dists)
    """
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_dists, bound_dists])
    all_indices = np.concatenate([np.arange(len(pair_dists)), np.arange(N) + len(pair_dists)])
    
    # Find k smallest
    k_actual = min(k, len(all_vals))
    sorted_indices = np.argsort(all_vals)[:k_actual]
    
    crit_indices = all_indices[sorted_indices]
    crit_vals = all_vals[sorted_indices]
    
    pair_crit = crit_indices < len(pair_dists)
    pair_idx = crit_indices[pair_crit]
    bound_idx = crit_indices[~pair_crit]
    
    return pair_idx, bound_idx, pair_dists, bound_dists

def _soft_min_obj_grad(x, beta, use_subset=False, subset_size=100):
    """
    Objective: -soft_min of constraints.
    If use_subset, only consider the most critical constraints.
    """
    p = x.reshape(-1, 2)
    
    if use_subset:
        pair_idx, bound_idx, pair_dists, bound_dists = _critical_subset(p, subset_size)
        # Re-fetch only critical pair distances
        iu = np.triu_indices(N, k=1)
        crit_pair_dists = pair_dists[pair_idx]
        crit_bound_dists = bound_dists[bound_idx]
    else:
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        crit_pair_dists = dists[iu] / 2.0
        norms = np.linalg.norm(p, axis=1)
        crit_bound_dists = 1.0 - norms
    
    # Log-sum-exp for soft min
    z_pairs = -beta * crit_pair_dists
    z_bounds = -beta * crit_bound_dists
    
    max_z = max(np.max(z_pairs), np.max(z_bounds))
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    log_sum_exp = max_z + np.log(sum_exp)
    val = log_sum_exp / beta
    
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp
    
    grad = np.zeros_like(p)
    
    # Pair contributions
    if use_subset:
        iu = np.triu_indices(N, k=1)
        i_idx, j_idx = iu[pair_idx]
    else:
        iu = np.triu_indices(N, k=1)
        i_idx, j_idx = iu
    
    safe_pair_dists = np.where(crit_pair_dists > 1e-12, crit_pair_dists, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    safe_norms = np.where(np.linalg.norm(p, axis=1) > 1e-12, np.linalg.norm(p, axis=1), 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    
    # Only add boundary gradient for critical boundaries
    if use_subset:
        grad[bound_idx] += w_bounds[:, np.newaxis] * bound_grad_vecs[bound_idx]
    else:
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return val, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Phase 1: Coarse annealing
    betas_coarse = [5.0, 15.0, 40.0, 100.0, 200.0]
    current = points.flatten()
    
    for beta in betas_coarse:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta, use_subset=False)[0],
                current,
                jac=lambda x: _soft_min_obj_grad(x, beta, use_subset=False)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 25, 'disp': False}
            )
            if time.time() < deadline:
                cand = res.x.reshape(-1, 2)
                s = _compute_score(cand)
                if s > best_score:
                    best_points = cand
                    best_score = s
                current = cand.flatten()
        except Exception:
            break
    
    # Phase 2: Fine annealing with critical subset
    betas_fine = [400.0, 800.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    subset_sizes = [100, 80, 60, 50, 40, 30, 20]
    
    for beta, ssz in zip(betas_fine, subset_sizes):
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta, use_subset=True, subset_size=ssz)[0],
                current,
                jac=lambda x: _soft_min_obj_grad(x, beta, use_subset=True, subset_size=ssz)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 30, 'disp': False}
            )
            if time.time() < deadline - 5:
                cand = res.x.reshape(-1, 2)
                s = _compute_score(cand)
                if s > best_score:
                    best_points = cand
                    best_score = s
                current = cand.flatten()
        except Exception:
            break
    
    # Phase 3: Very high beta polish on critical subset
    for beta in [50000.0, 100000.0]:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta, use_subset=True, subset_size=15)[0],
                current,
                jac=lambda x: _soft_min_obj_grad(x, beta, use_subset=True, subset_size=15)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 20, 'disp': False}
            )
            if time.time() < deadline - 1:
                cand = res.x.reshape(-1, 2)
                s = _compute_score(cand)
                if s > best_score:
                    best_points = cand
                    best_score = s
                    current = cand.flatten()
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
