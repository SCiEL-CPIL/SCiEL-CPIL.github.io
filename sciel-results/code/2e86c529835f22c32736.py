"""Rewritten program for packing 50 circles in a unit circle using soft-min L-BFGS-B optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """
    Compute the packing score for given centers.
    Score = min(min_pair_dist / 2, min_bound_dist)
    """
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min_objective_and_grad(x, beta):
    """
    Soft-min surrogate for the packing problem.
    
    We want to maximize: min( min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|) )
    
    Using soft-min: f(x) = -(1/beta) * log(sum(exp(-beta * d_k)))
    where d_k are the constraint distances (pair distances/2 and boundary distances).
    
    x is a flat array of shape (2*N,).
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft-min: -(1/beta) * log(sum(exp(-beta * d)))
    # For numerical stability, subtract max
    max_d = np.max(all_dists)
    shifted = -beta * (all_dists - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(1.0 / beta) * log_sum_exp + max_d
    
    # Gradient computation
    # d(soft_min)/dx = (1/beta) * sum_k (exp(-beta*d_k) / sum_j exp(-beta*d_j)) * d(-d_k)/dx
    # = (1/beta) * sum_k (exp(-beta*d_k) / sum_j exp(-beta*d_j)) * (-dd_k/dx)
    # = -sum_k w_k * (dd_k/dx)  where w_k = exp(-beta*d_k)/sum_j exp(-beta*d_j)
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    weights = weights / beta  # This gives (1/beta) * softmax weights
    
    grad = np.zeros_like(x)
    
    # Pair distance gradients
    # d(|c_i - c_j|/2)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(|c_i - c_j|/2)/dc_j = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # Vectorized gradient for pairs
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = weights[idx]
        # Gradient contribution to center i
        grad[2*i] += -w * (centers[i, 0] - centers[j, 0]) / d
        grad[2*i+1] += -w * (centers[i, 1] - centers[j, 1]) / d
        # Gradient contribution to center j
        grad[2*j] += -w * (centers[j, 0] - centers[i, 0]) / d
        grad[2*j+1] += -w * (centers[j, 1] - centers[i, 1]) / d
    
    # Boundary distance gradients
    # d(1 - |c_i|)/dc_i = -c_i / |c_i|
    for i in range(N):
        norm_i = norms[i]
        if norm_i < 1e-10:
            continue
        w = weights[len(pair_dists) + i]
        grad[2*i] += -w * (-centers[i, 0] / norm_i)
        grad[2*i+1] += -w * (-centers[i, 1] / norm_i)
    
    # Note: we're minimizing -soft_min, so the objective is -soft_min_val
    # and gradient is -grad (since d(-f)/dx = -df/dx)
    return -soft_min_val, -grad

def _soft_min_objective_and_grad_vectorized(x, beta):
    """
    Vectorized soft-min surrogate with gradient using np.add.at for efficiency.
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    n_pairs = len(pair_dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, bound_dists])
    n_total = len(all_dists)
    
    # Soft-min with numerical stability
    max_d = np.max(all_dists)
    shifted = -beta * (all_dists - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(1.0 / beta) * log_sum_exp + max_d
    
    # Weights: w_k = exp(-beta*d_k) / (beta * sum_j exp(-beta*d_j))
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / (beta * sum_exp)
    
    grad = np.zeros(2 * N)
    
    # Pair distance gradients using vectorized operations
    # For each pair (i,j): 
    #   grad contribution to i: -w * (c_i - c_j) / |c_i - c_j|
    #   grad contribution to j: -w * (c_j - c_i) / |c_i - c_j|
    
    i_idx = iu[0]
    j_idx = iu[1]
    d_vals = dists[i_idx, j_idx]
    
    # Avoid division by zero
    valid = d_vals > 1e-10
    if np.any(valid):
        w_valid = weights[:n_pairs][valid]
        d_valid = d_vals[valid]
        i_valid = i_idx[valid]
        j_valid = j_idx[valid]
        
        # Compute unit vectors
        diff_ij = centers[i_valid] - centers[j_valid]
        unit_vecs = diff_ij / d_valid[:, np.newaxis]
        
        # Contributions
        contrib_i = -w_valid[:, np.newaxis] * unit_vecs
        contrib_j = -w_valid[:, np.newaxis] * unit_vecs
        
        # Use np.add.at for accumulation
        np.add.at(grad, 2 * i_valid, contrib_i[:, 0])
        np.add.at(grad, 2 * i_valid + 1, contrib_i[:, 1])
        np.add.at(grad, 2 * j_valid, contrib_j[:, 0])
        np.add.at(grad, 2 * j_valid + 1, contrib_j[:, 1])
    
    # Boundary distance gradients
    # d(1 - |c_i|)/dc_i = -c_i / |c_i|
    valid_bound = norms > 1e-10
    if np.any(valid_bound):
        w_bound = weights[n_pairs:][valid_bound]
        c_valid = centers[valid_bound]
        norm_valid = norms[valid_bound]
        
        # Gradient: -w * (-c_i / |c_i|) = w * c_i / |c_i|
        contrib = w_bound[:, np.newaxis] * c_valid / norm_valid[:, np.newaxis]
        
        idx = np.where(valid_bound)[0]
        np.add.at(grad, 2 * idx, contrib[:, 0])
        np.add.at(grad, 2 * idx + 1, contrib[:, 1])
    
    return -soft_min_val, -grad

def _get_hex_lattice_init(n, seed=42):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) == 0:
        return _get_random_init(n, seed=seed)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_init(n, seed=seed)
    
    # Sort by distance to center to pick central ones
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_init(n, seed=42):
    """
    Generate random initialization inside the unit circle.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.95**2:
                centers[i] = [x, y]
                break
    return centers

def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization that approximates the optimal packing structure.
    """
    rng = np.random.default_rng(seed)
    centers = []
    
    # Center
    centers.append([0.0, 0.0])
    
    # Rings with approximate positions based on known packing structure
    # For 50 circles: 1 + 6 + 12 + 12 + 9 + 10 = 50
    ring_configs = [
        (6, 0.25),
        (12, 0.50),
        (12, 0.75),
        (9, 0.88),
        (10, 0.95),
    ]
    
    count = 1
    for count_in_ring, radius in ring_configs:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        # Add small random perturbation to break symmetry
        angles += rng.uniform(-0.1, 0.1, count_in_ring)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
    
    return np.array(centers[:n])

def _optimize_with_lbfgsb(x0, beta_start, beta_end, max_iters, time_budget, start_time, tol=1e-8):
    """
    Run L-BFGS-B optimization with adaptive beta (soft-min steepness).
    """
    x = x0.copy()
    current_beta = beta_start
    
    # We'll run multiple phases with increasing beta
    n_phases = 5
    betas = np.linspace(beta_start, beta_end, n_phases)
    
    for phase, beta in enumerate(betas):
        if time.time() - start_time > time_budget:
            break
        
        # Run L-BFGS-B with current beta
        def obj_grad(x_flat):
            val, grad = _soft_min_objective_and_grad_vectorized(x_flat, beta)
            return val, grad
        
        result = minimize(
            obj_grad,
            x,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iters,
                'ftol': tol,
                'gtol': tol,
                'maxls': 20,
            }
        )
        
        if result.x is not None:
            x = result.x
        
        # Increase beta for next phase
        current_beta = beta
    
    return x

def _polish_precise(x, time_budget, start_time):
    """
    High-precision polish with very high beta and tight tolerance.
    """
    x = x.copy()
    
    # Final phase with very high beta for precision
    beta = 10000.0
    tol = 1e-10
    
    def obj_grad(x_flat):
        val, grad = _soft_min_objective_and_grad_vectorized(x_flat, beta)
        return val, grad
    
    try:
        result = minimize(
            obj_grad,
            x,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': tol,
                'gtol': tol,
                'maxls': 20,
            }
        )
        if result.x is not None:
            x = result.x
    except Exception:
        pass
    
    return x

def _project_inside(x):
    """
    Project points inside the unit circle.
    """
    centers = x.reshape(N, 2)
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        centers = centers.copy()
        centers[mask] *= 0.999 / norms[mask]
    return centers.reshape(-1)

def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Multiple diverse initializations
    initializations = [
        _get_hex_lattice_init(N, seed=42),
        _get_ring_init(N, seed=1),
        _get_ring_init(N, seed=42),
        _get_random_init(N, seed=1),
        _get_random_init(N, seed=99),
        _get_hex_lattice_init(N, seed=123),
    ]
    
    # Time budget allocation
    # Total: ~145s
    # Each init: ~20s for main optimization + ~5s for polish
    # 6 inits * 25s = 150s, too much
    # Let's do 4 inits with 30s each + final polish
    
    n_inits = min(4, len(initializations))
    per_init_time = (DEADLINE - 10) / n_inits  # Reserve 10s for final polish
    
    for i in range(n_inits):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        init = initializations[i].copy()
        # Ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
        
        x0 = init.reshape(-1)
        
        # Main optimization phase
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining <= 0:
            break
        
        opt_time = min(per_init_time, remaining)
        
        x_opt = _optimize_with_lbfgsb(
            x0,
            beta_start=10.0,
            beta_end=1000.0,
            max_iters=200,
            time_budget=opt_time,
            start_time=start_time
        )
        
        # Project inside
        x_opt = _project_inside(x_opt)
        
        # Polish phase
        polish_time = min(5.0, DEADLINE - 5 - (time.time() - start_time))
        if polish_time > 0:
            x_opt = _polish_precise(x_opt, polish_time, start_time)
            x_opt = _project_inside(x_opt)
        
        # Evaluate
        centers = x_opt.reshape(N, 2)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final polish on the best if time remains
    if best_centers is not None and time.time() - start_time < DEADLINE - 2:
        final_time = DEADLINE - 2 - (time.time() - start_time)
        if final_time > 0:
            x_best = best_centers.reshape(-1)
            x_best = _polish_precise(x_best, final_time, start_time)
            x_best = _project_inside(x_best)
            final_centers = x_best.reshape(N, 2)
            final_score = _compute_score(final_centers)
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = initializations[0]
        
    return best_centers
