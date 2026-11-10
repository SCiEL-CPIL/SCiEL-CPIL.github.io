"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140  # 10s safety margin from 150s limit

def _hex_lattice_init():
    """Generate initial positions using a hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Select N points closest to center to keep them well-distributed
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        spacing *= 0.98

def _calc_score(C):
    """Calculate the effective radius for a given set of centers."""
    C = np.asarray(C).reshape(-1, 2)
    if C.shape[0] != N:
        return 0.0
    # Check for non-finite or out-of-bound
    if not np.all(np.isfinite(C)):
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(C, axis=1)
    r_bound = 1.0 - np.max(norms)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _soft_min_objective(x, beta):
    """
    Smooth surrogate for maximizing min(r_bound, r_pair).
    We minimize the negative of a soft-min.
    
    The constraints are:
    - For each i: 1 - ||c_i|| >= r  (boundary)
    - For each i<j: ||c_i - c_j||/2 >= r  (pairwise)
    
    We want to maximize r such that all constraints hold.
    Using a soft-min: r_soft = (1/beta) * log(sum(exp(-beta * g_k)))
    where g_k are the constraint functions (larger is better).
    
    Actually, let's think differently. We want to maximize the minimum of:
    - g_i = 1 - ||c_i|| for i=1..N
    - h_ij = ||c_i - c_j||/2 for i<j
    
    The soft-min of a set of values v_k is:
    softmin(v) = -logsumexp(-beta * v_k) / beta
    
    We want to maximize softmin, so we minimize -softmin.
    """
    C = x.reshape(N, 2)
    
    # Boundary constraints: g_i = 1 - ||c_i||
    norms = np.linalg.norm(C, axis=1)
    g_bound = 1.0 - norms  # shape (N,)
    
    # Pairwise constraints: h_ij = ||c_i - c_j|| / 2
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = C[i_idx] - C[j_idx]
    d_pairs = np.linalg.norm(diff, axis=1)
    g_pair = d_pairs / 2.0  # shape (N*(N-1)/2,)
    
    # Combine all constraints
    g_all = np.concatenate([g_bound, g_pair])
    
    # Soft-min: -logsumexp(-beta * g) / beta
    # For numerical stability, use the max trick
    max_g = np.max(g_all)
    # If max_g is very negative, we're in trouble
    exp_terms = np.exp(-beta * (g_all - max_g))
    logsumexp_val = np.log(np.sum(exp_terms)) + (-beta * max_g)
    soft_min_val = -logsumexp_val / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective."""
    C = x.reshape(N, 2)
    
    # Boundary constraints: g_i = 1 - ||c_i||
    norms = np.linalg.norm(C, axis=1)
    g_bound = 1.0 - norms
    
    # Pairwise constraints: h_ij = ||c_i - c_j|| / 2
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = C[i_idx] - C[j_idx]
    d_pairs = np.linalg.norm(diff, axis=1)
    g_pair = d_pairs / 2.0
    
    # Combine all constraints
    g_all = np.concatenate([g_bound, g_pair])
    n_constraints = len(g_all)
    
    # Soft-min: -logsumexp(-beta * g) / beta
    max_g = np.max(g_all)
    exp_terms = np.exp(-beta * (g_all - max_g))
    logsumexp_val = np.log(np.sum(exp_terms)) + (-beta * max_g)
    
    # Weights: w_k = exp(-beta * g_k) / sum(exp(-beta * g_k))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft_min w.r.t. g_k: -w_k
    # So gradient of -soft_min w.r.t. g_k: w_k
    
    # Now chain rule: d(-soft_min)/dx = sum_k w_k * d(g_k)/dx
    
    grad_x = np.zeros_like(C)
    
    # Boundary part: g_i = 1 - ||c_i||
    # d(g_i)/d(c_i) = -c_i / ||c_i||
    # Contribution to grad_x: sum_{i in boundary} w_i * (-c_i / ||c_i||)
    # But w_i here refers to the weight of the i-th boundary constraint
    w_bound = weights[:N]
    safe_norms = norms.copy()
    safe_norms[safe_norms == 0] = 1e-10
    grad_x -= w_bound[:, np.newaxis] * (C / safe_norms[:, np.newaxis])
    
    # Pairwise part: g_ij = ||c_i - c_j|| / 2
    # d(g_ij)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d(g_ij)/d(c_j) = (c_j - c_i) / (2 * ||c_i - c_j||) = -(c_i - c_j) / (2 * ||c_i - c_j||)
    w_pair = weights[N:]
    n_pairs = len(i_idx)
    
    # For each pair, accumulate contributions
    safe_d = d_pairs.copy()
    safe_d[safe_d == 0] = 1e-10
    
    # Gradient contribution for each pair
    # d_vec = c_i - c_j
    # d(g_ij)/d(c_i) = d_vec / (2 * d_ij)
    # d(g_ij)/d(c_j) = -d_vec / (2 * d_ij)
    
    grad_pair = np.zeros_like(C)
    for k in range(n_pairs):
        gi = i_idx[k]
        gj = j_idx[k]
        w = w_pair[k]
        dv = diff[k]
        factor = w / (2.0 * safe_d[k])
        grad_pair[gi] += factor * dv
        grad_pair[gj] -= factor * dv
    
    grad_x += grad_pair
    
    return grad_x.flatten()

def _objective_and_grad(x, r_target):
    """
    Calculate the violation cost and gradient for a given radius target.
    x: flattened array of centers (2*N)
    r_target: target radius
    """
    C = x.reshape(N, 2)
    
    # 1. Boundary Violations
    norms = np.linalg.norm(C, axis=1)
    # Violation if norm > 1 - r_target
    # v_i = max(0, norm_i - (1 - r_target))
    # Cost = sum v_i^2
    max_norm_allowed = 1.0 - r_target
    over_bound = norms - max_norm_allowed
    pos_over_bound = np.maximum(0.0, over_bound)
    
    cost_bound = np.sum(pos_over_bound ** 2)
    
    # Gradient for boundary part
    # d/dC_i sum(max(0, ||C_i|| - M)^2)
    # = 2 * max(0, ||C_i|| - M) * (C_i / ||C_i||)
    # If ||C_i|| is 0, gradient is 0 (or undefined, treat as 0)
    grad_bound = np.zeros_like(C)
    mask = pos_over_bound > 0
    if np.any(mask):
        # C[mask] / norms[mask]
        # Avoid division by zero
        safe_norms = norms[mask]
        safe_norms[safe_norms == 0] = 1e-10
        grad_bound[mask] = 2.0 * pos_over_bound[mask, np.newaxis] * (C[mask] / safe_norms[:, np.newaxis])

    # 2. Pairwise Distance Violations
    # For i < j, d_ij < 2 * r_target is a violation
    # v_ij = max(0, 2*r_target - d_ij)
    # Cost = sum v_ij^2
    
    # Compute pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    min_dist_req = 2.0 * r_target
    # We only care about i < j to avoid double counting in sum, 
    # but for gradient we need to account for both i and j contributions.
    # Sum over all i != j of max(0, R - d_ij)^2 / 2 ? 
    # The objective is sum_{i<j} v_ij^2.
    # Let's compute the upper triangle.
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_pairs = dists[i_idx, j_idx]
    
    over_dist = min_dist_req - d_pairs
    pos_over_dist = np.maximum(0.0, over_dist)
    
    cost_pair = np.sum(pos_over_dist ** 2)
    
    # Gradient for pairwise part
    # Term for pair (i,j): v^2 = (R - d)^2
    # d/dc_i v^2 = 2 * v * (-(c_i - c_j) / d)
    # d/dc_j v^2 = 2 * v * (-(c_j - c_i) / d)
    
    grad_pair = np.zeros_like(C)
    mask_pair = pos_over_dist > 0
    if np.any(mask_pair):
        valid_i = i_idx[mask_pair]
        valid_j = j_idx[mask_pair]
        valid_d = d_pairs[mask_pair]
        valid_v = pos_over_dist[mask_pair]
        
        # Avoid division by zero
        valid_d[valid_d == 0] = 1e-10
        
        # Gradient vector for each pair
        # d_vec = c_i - c_j
        d_vec = C[valid_i] - C[valid_j]
        
        # Factor: 2 * v / d
        factor = (2.0 * valid_v / valid_d)
        
        # Grad for i: -factor * d_vec
        # Grad for j: +factor * d_vec
        
        # Accumulate
        for k in range(len(valid_i)):
            gi = valid_i[k]
            gj = valid_j[k]
            f = factor[k]
            dv = d_vec[k]
            grad_pair[gi] -= f * dv
            grad_pair[gj] += f * dv

    total_cost = cost_bound + cost_pair
    total_grad = (grad_bound + grad_pair).flatten()
    
    return total_cost, total_grad

def _polish(C_init, max_time=10.0):
    """
    High-precision polish pass using the soft-min objective.
    Starts from C_init and refines with increasing beta for sharper approximation.
    """
    C = C_init.copy()
    best_C = C.copy()
    best_score = _calc_score(best_C)
    
    # Start with moderate beta, increase for sharper approximation
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        def fun(x):
            return _soft_min_objective(x, beta)
        
        def jac(x):
            return _soft_min_grad(x, beta)
        
        try:
            res = minimize(
                fun,
                C.flatten(),
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            
            C_new = res.x.reshape(N, 2)
            score_new = _calc_score(C_new)
            
            if score_new > best_score:
                best_score = score_new
                best_C = C_new.copy()
                C = C_new.copy()
            else:
                # Keep the old C if it's better
                C = best_C.copy()
                
        except Exception:
            continue
    
    return best_C, best_score

def construct():
    # Initial guess from hex lattice
    C_init = _hex_lattice_init()
    best_C = C_init.copy()
    best_score = _calc_score(best_C)
    
    # Target radius search using binary search with constraint violation
    # Known best is ~0.125825
    r_low = best_score
    r_high = 0.1265  # Slightly above known best
    
    # If best_score is already high, narrow the range
    if best_score > 0.125:
        r_low = best_score
        r_high = best_score + 0.002

    # Binary search for the maximum feasible radius
    num_iters = 8  # Limit iterations to fit in time
    
    for it in range(num_iters):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Objective function for minimize
        def fun(x):
            return _objective_and_grad(x, r_mid)[0]
        
        def jac(x):
            return _objective_and_grad(x, r_mid)[1]
        
        try:
            res = minimize(
                fun, 
                best_C.flatten(),  # Use best found as warm start
                jac=jac, 
                method='L-BFGS-B', 
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 50}
            )
            
            C_new = res.x.reshape(N, 2)
            score_new = _calc_score(C_new)
            
            # Update best
            if score_new > best_score:
                best_score = score_new
                best_C = C_new.copy()
            
            # Check if feasible for r_mid
            if score_new >= r_mid - 1e-6:
                r_low = r_mid
            else:
                r_high = r_mid
                
        except Exception:
            break
    
    # Final high-precision polish pass
    if time.time() - START_TIME < TIME_LIMIT - 5:
        polished_C, polished_score = _polish(best_C)
        if polished_score > best_score:
            best_C = polished_C
            best_score = polished_score
    
    return best_C

# EVOLVE-BLOCK-END
