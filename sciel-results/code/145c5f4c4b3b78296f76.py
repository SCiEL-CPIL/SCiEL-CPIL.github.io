"""Improved packer: smooth surrogate optimization with L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _score_fast(points, dist):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - np.max(norm)
    if r_bound < 0.0:
        return 0.0
    # minimum pairwise half-distance
    # Note: dist has 0 on diagonal
    r_pair = 0.5 * np.min(dist)
    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _soft_min_objective(x, beta):
    """
    Soft-min surrogate for the packing objective.
    x: flattened array of shape (2*N,) representing N points in 2D.
    beta: temperature parameter for soft-min.
    
    Returns:
        objective value (negative of soft-min radius, since we minimize)
        gradient
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    r_bound = 1.0 - norms
    
    # For pairwise: we want min of dists[i,j] for i<j
    # For boundary: we want min of r_bound
    # Combine: create a matrix of all constraints
    # Use soft-min via log-sum-exp
    
    # Pairwise distances (upper triangle only)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # All constraints: pairwise half-distances and boundary distances
    all_constraints = np.concatenate([0.5 * pair_dists, r_bound])
    
    # Soft-min: -log(sum(exp(-beta * c))) / beta
    # For numerical stability, use max trick
    c_max = np.max(all_constraints)
    shifted = all_constraints - c_max
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * c_max)
    soft_min_val = -log_sum_exp / beta
    
    # Gradient computation
    # d(soft_min)/dx = -1/beta * sum(exp(-beta*c_i) * dc_i/dx) / sum(exp(-beta*c_i))
    
    # Compute weights
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Pairwise contribution
    # For each pair (i,j), constraint c_ij = 0.5 * ||p_i - p_j||
    # dc_ij/dp_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
    # dc_ij/dp_j = 0.5 * (p_j - p_i) / ||p_i - p_j||
    
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = weights[k] * beta  # weight for this constraint
        if w < 1e-12:
            continue
        d_ij = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d_ij)
        if dist_ij > 1e-12:
            grad[2*i:2*i+2] += w * 0.5 * d_ij / dist_ij
            grad[2*j:2*j+2] += w * 0.5 * (-d_ij) / dist_ij
    
    # Boundary contribution
    for i in range(N):
        w = weights[len(iu[0]) + i] * beta
        if w < 1e-12:
            continue
        norm_i = np.linalg.norm(pts[i])
        if norm_i > 1e-12:
            # dc/dp = -p/||p||
            grad[2*i:2*i+2] += w * (-pts[i] / norm_i)
    
    # Negate since we want to maximize soft_min, so minimize -soft_min
    obj = -soft_min_val
    grad = -grad
    
    return obj, grad


def _soft_min_objective_fast(x, beta):
    """
    Vectorized soft-min objective for better performance.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    r_bound = 1.0 - norms
    
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = 0.5 * dists[iu]
    
    # All constraints
    all_constraints = np.concatenate([pair_dists, r_bound])
    
    # Soft-min
    c_max = np.max(all_constraints)
    shifted = all_constraints - c_max
    exp_vals = np.exp(-beta * shifted)
    log_sum_exp = np.log(np.sum(exp_vals)) + (-beta * c_max)
    soft_min_val = -log_sum_exp / beta
    
    # Gradient
    weights = exp_vals / np.sum(exp_vals)
    grad = np.zeros_like(x)
    
    # Pairwise gradient (vectorized)
    # For pair (i,j): dc/dp_i = 0.5*(p_i-p_j)/d, dc/dp_j = 0.5*(p_j-p_i)/d
    d_ij = dists[iu]
    valid = d_ij > 1e-12
    w_pair = weights[:len(iu[0])]
    
    # Accumulate gradient for each point
    # Use scatter-add approach
    grad_pairs = np.zeros((N, 2))
    for k in np.where(valid)[0]:
        i, j = iu[0][k], iu[1][k]
        w = w_pair[k] * beta
        if w < 1e-12:
            continue
        d = d_ij[k]
        dir_ij = (pts[i] - pts[j]) / d
        grad_pairs[i] += 0.5 * w * dir_ij
        grad_pairs[j] -= 0.5 * w * dir_ij
    
    grad[:2*N:2] = grad_pairs[:, 0]
    grad[1:2*N:2] = grad_pairs[:, 1]
    
    # Boundary gradient
    w_bound = weights[len(iu[0]):]
    valid_bound = norms > 1e-12
    for i in np.where(valid_bound)[0]:
        w = w_bound[i] * beta
        if w < 1e-12:
            continue
        grad_pairs[i] += w * (-pts[i] / norms[i])
    
    grad[:2*N:2] = grad_pairs[:, 0]
    grad[1:2*N:2] = grad_pairs[:, 1]
    
    obj = -soft_min_val
    grad = -grad
    
    return obj, grad


def _polish_lbfgs(points, t_budget, beta_start=10.0, beta_end=100000.0):
    """
    Polish using L-BFGS-B with increasing beta (soft-min temperature).
    """
    start = time.time()
    best = points.copy()
    best_r = _score_fast(best, _init_dist(best))
    
    # Start with moderate beta, increase for finer optimization
    betas = [beta_start, 50.0, 200.0, 500.0, 2000.0, 20000.0, beta_end]
    
    cur = points.copy()
    
    for beta in betas:
        if time.time() - start > t_budget - 5.0:
            break
        
        x0 = cur.reshape(-1)
        
        # Use L-BFGS-B with gradient
        result = minimize(
            _soft_min_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 50,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20
            }
        )
        
        cur = result.x.reshape(N, 2)
        
        # Ensure points are inside the container
        norms = np.linalg.norm(cur, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            cur[mask] = cur[mask] / norms[mask, None] * (1.0 - 1e-9)
        
        r = _score_fast(cur, _init_dist(cur))
        if r > best_r:
            best_r = r
            best = cur.copy()
    
    return best, best_r


def _init_dist(points):
    """Compute initial pairwise distance matrix."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score_fast(best, _init_dist(best))

    # Phase 1: Multi-start with gradient-based optimization
    # Try multiple initializations with different jitter levels
    jitter_levels = [0.0, 0.005, 0.01, 0.02, 0.03]
    
    for jitter_level in jitter_levels:
        if time.time() - t0 > DEADLINE - 30.0:
            break
        
        if jitter_level == 0.0:
            cand = best.copy()
        else:
            jitter = jitter_level * rng.standard_normal((N, 2))
            cand = best + jitter
            # Clip inside container
            rad = np.linalg.norm(cand, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
        
        t_left = DEADLINE - 30.0 - (time.time() - t0)
        if t_left < 10.0:
            break
        
        t_alloc = min(t_left - 5.0, 20.0)
        cand, r = _polish_lbfgs(cand, t_alloc)
        
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: Final high-precision polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        best, best_r = _polish_lbfgs(best, t_left - 5.0, beta_start=100.0, beta_end=200000.0)

    # final safety: ensure valid output
    final_r = _score_fast(best, _init_dist(best))
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
