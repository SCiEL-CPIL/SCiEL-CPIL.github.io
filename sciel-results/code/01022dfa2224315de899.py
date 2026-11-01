"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the packing score for given centers."""
    if not np.all(np.isfinite(centers)):
        return -np.inf
    n = len(centers)
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

def _soft_min_objective(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    Constraints: pairwise distances / 2 and boundary distances (1 - norm)
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    # To avoid overflow, we shift by the min
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    soft_min_val = m - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min_val

def _soft_min_gradient(params, n, beta):
    """
    Gradient of the soft min objective.
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    vals = np.concatenate([pair_dists, bound_dists])
    
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    grad = np.zeros_like(centers)
    
    # Gradient with respect to pair distances
    # d(pair_dists)/d(centers) involves (c_i - c_j) / |c_i - c_j|
    # Note: pair_dists = |c_i - c_j| / 2
    # d(|c_i - c_j|)/d c_i = (c_i - c_j) / |c_i - c_j|
    # So d(pair_dists)/d c_i = 0.5 * (c_i - c_j) / |c_i - c_j|
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            # Objective is -soft_min. 
            # d(-soft_min)/d c = - d(soft_min)/d c
            # d(soft_min)/d c = sum(weights * d(val)/d c)
            # So grad = - sum(weights * d(val)/d c)
            
            # d(pair_dists)/d c_i = 0.5 * vec
            # d(pair_dists)/d c_j = -0.5 * vec
            
            grad[i] -= w * 0.5 * vec
            grad[j] += w * 0.5 * vec

    # Gradient with respect to boundary distances
    # bound_dists = 1 - |c_i|
    # d(bound_dists)/d c_i = - c_i / |c_i|
    
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm > 1e-8:
            vec = centers[i] / norm
            # d(bound_dists)/d c_i = -vec
            # grad = - w * (-vec) = w * vec
            grad[i] += w * vec
            
    return grad.flatten()

def _objective_and_grad(params, n, beta):
    obj = _soft_min_objective(params, n, beta)
    grad = _soft_min_gradient(params, n, beta)
    return obj, grad

def _greedy_init(n, seed=42):
    """
    Greedy initialization: place points one by one to maximize min distance.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    
    # First point at center
    centers[0] = [0.0, 0.0]
    
    for i in range(1, n):
        best_score = -np.inf
        best_pos = None
        
        # Sample many candidates
        for _ in range(2000):
            # Sample uniformly in circle
            r = np.sqrt(rng.uniform(0, 1))
            theta = rng.uniform(0, 2 * np.pi)
            pos = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            # Calculate min distance to existing points and boundary
            dists_to_points = np.linalg.norm(centers[:i] - pos, axis=1)
            min_dist_point = np.min(dists_to_points) / 2.0 if i > 0 else np.inf
            dist_to_boundary = 1.0 - np.linalg.norm(pos)
            
            score = min(min_dist_point, dist_to_boundary)
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        centers[i] = best_pos
        
    return centers

def _random_init(n, seed=42):
    """
    Random initialization.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        centers[i] = [r * np.cos(theta), r * np.sin(theta)]
    return centers

def _polish_gradient_ascent(centers, start_time, max_iters=5000, step_size=0.005):
    """
    Direct gradient ascent on the minimum constraint.
    Identifies the tightest constraint and moves points to increase it.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        # Determine bottleneck
        # r = min(min_pair_dist/2, min_bound_dist)
        
        step = step_size
        # Adaptive step: decay slightly
        step *= (1.0 - it / (max_iters * 1.5))
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j: i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                # Move i and j apart
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                # Move i towards center
                current[i] -= vec * step
        
        # Ensure no points go out of bounds (safety check)
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / new_norms[mask]
            
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            
    return current, best_score

def _optimize_single(centers, start_time):
    """
    Multi-stage optimization for a single initialization.
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B with soft min
    betas = [20.0, 50.0, 100.0, 200.0, 500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        try:
            res = minimize(
                _objective_and_grad, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    current = candidate
        except Exception:
            pass
            
    # Stage 2: Gradient Ascent Polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_gradient_ascent(current, start_time, max_iters=3000, step_size=0.003)
        
    return current

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _greedy_init(N, seed=1),
        _greedy_init(N, seed=42),
        _random_init(N, seed=100),
        _random_init(N, seed=200),
        _random_init(N, seed=300),
    ]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        optimized = _optimize_single(init, start_time)
        
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_gradient_ascent(best_centers, start_time, max_iters=5000, step_size=0.001)
        
    return best_centers
# EVOLVE-BLOCK-END
