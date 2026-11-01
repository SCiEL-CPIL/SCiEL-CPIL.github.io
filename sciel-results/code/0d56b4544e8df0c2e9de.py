"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize
import time

N = 50

# EVOLVE-BLOCK-START

def _hex_lattice_init(n):
    """Generate initial points on a hexagonal lattice."""
    points = []
    # Approximate radius of the container
    R = 1.0
    # Estimate spacing
    # Area of unit circle = pi. Area of hex cell ~ 2.598 * r^2.
    # N circles, area ~ N * pi * r_c^2.
    # r_c ~ sqrt(pi / (N * 2.598 * pi)) = 1 / sqrt(2.598 * N) ~ 1 / sqrt(2.6 * 50) ~ 1/11.4 ~ 0.087
    # Diameter ~ 0.17.
    # Let's just generate a dense lattice and pick the first N inside.
    spacing = 0.1
    k = 15
    pts = []
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            x = i * spacing + (j % 2) * spacing / 2.0
            y = j * spacing * np.sqrt(3) / 2.0
            if x*x + y*y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < n:
        # Fallback to random
        angles = np.random.uniform(0, 2*np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Sort by distance from center to pick central ones? Or just first N.
        # Better to pick N points that are most "spread out" or just random subset?
        # For initialization, a random subset of the lattice is fine to break symmetry.
        indices = np.random.choice(len(pts), n, replace=False)
        pts = pts[indices]
    return pts

def _soft_min_obj_and_grad(x, beta):
    """
    Objective: Maximize soft-min of distances.
    x: shape (N, 2)
    beta: temperature
    Returns: -soft_min_value, -gradient
    """
    n = len(x)
    # Distances between all pairs
    # d_ij = ||x_i - x_j||
    # We want min d_ij.
    # Soft min: -1/beta * log(sum exp(-beta * d_ij))
    
    # Calculate all pairwise distances
    # x is (N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Boundary distances: 1 - ||x_i||
    norms = np.linalg.norm(x, axis=1)
    b_dists = 1.0 - norms
    
    # We combine pair distances and boundary distances into one set of values
    # We only need i < j for pairs to avoid double counting, but for the sum it doesn't matter much if we are consistent.
    # Let's use i < j to save computation.
    iu = np.triu_indices(n, 1)
    d_pairs = dists[iu]
    
    all_dists = np.concatenate([d_pairs, b_dists])
    
    # Soft min calculation
    # To avoid overflow/underflow, use log-sum-exp trick
    # log(sum(exp(-beta * d)))
    # = log(sum(exp(-beta * (d - min_d)))) + (-beta * min_d)
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    log_sum_exp = np.log(np.sum(exp_terms)) - beta * min_d
    soft_min_val = -1.0 / beta * log_sum_exp
    
    # Gradient
    # d/dx soft_min = -1/beta * d/dx log_sum_exp
    # d/dx log(sum exp(-beta d_k)) = sum(exp(-beta d_k) * (-beta d_k/dx)) / sum(exp(-beta d_k))
    # weights = exp_terms / sum(exp_terms)
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of -soft_min = 1/beta * d/dx log_sum_exp
    # grad_log_sum = -beta * sum(weights_k * grad d_k)
    # So grad_obj = 1/beta * (-beta * sum(weights_k * grad d_k)) = -sum(weights_k * grad d_k)
    # Wait, we are maximizing soft_min. So we minimize -soft_min.
    # Let F = -soft_min. We want grad F.
    # F = 1/beta * log_sum_exp
    # grad F = 1/beta * (-beta * sum(weights_k * grad d_k)) = -sum(weights_k * grad d_k)
    
    grad = np.zeros_like(x)
    
    # Pair distances gradient
    # d_{ij} = ||x_i - x_j||
    # grad_{x_i} d_{ij} = (x_i - x_j) / d_{ij}
    # grad_{x_j} d_{ij} = (x_j - x_i) / d_{ij}
    
    # We have weights for pairs in weights[:len(d_pairs)]
    w_pairs = weights[:len(d_pairs)]
    # d_pairs corresponds to iu[0], iu[1]
    i_idx, j_idx = iu
    # Accumulate gradients
    # For each pair (i, j) with weight w:
    # grad_x_i += -w * (x_i - x_j) / d_ij
    # grad_x_j += -w * (x_j - x_i) / d_ij
    
    # Vectorized accumulation
    # diff_ij = x_i - x_j
    diff_ij = x[i_idx] - x[j_idx]
    # d_ij is d_pairs
    # unit vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff_ij / d_pairs[:, np.newaxis]
        # Handle zeros
        unit_vecs[np.isnan(unit_vecs)] = 0
    
    # grad contribution to x_i is -w * unit_vec
    # grad contribution to x_j is -w * (-unit_vec) = w * unit_vec
    
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * unit_vecs)
    
    # Boundary distances gradient
    # b_i = 1 - ||x_i||
    # grad b_i = -x_i / ||x_i||
    w_b = weights[len(d_pairs):]
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_norms = x / norms[:, np.newaxis]
        unit_norms[np.isnan(unit_norms)] = 0
    # grad_x_i for b_i is -x_i / ||x_i||
    # Contribution to grad F: -w_b * grad b_i = -w_b * (-x_i / ||x_i||) = w_b * x_i / ||x_i||
    grad += w_b[:, np.newaxis] * unit_norms
    
    # We are minimizing -soft_min, so return -soft_min_val and grad
    return -soft_min_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess
    x0 = _hex_lattice_init(N)
    
    # Annealing schedule for beta
    betas = [10, 50, 100, 200, 500, 1000]
    
    best_x = x0
    best_val = -np.inf
    
    # Evaluate initial
    val, _ = _soft_min_obj_and_grad(x0, betas[0])
    # Note: _soft_min_obj_and_grad returns -soft_min. So lower is better for minimization.
    # We want to maximize soft_min, so we minimize -soft_min.
    # best_val should be the minimum of the returned value.
    best_val = val
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = scipy.optimize.minimize(
                _soft_min_obj_and_grad, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x0 = res.x
            val, _ = _soft_min_obj_and_grad(x0, beta)
            if val < best_val:
                best_val = val
                best_x = x0.copy()
        except Exception:
            pass
            
    # Final check and cleanup
    # Ensure points are inside the circle
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms <= 1.0
    if not np.all(mask):
        # Project points back if they are outside (shouldn't happen with L-BFGS-B if we had bounds, but we didn't use bounds)
        # Actually, L-BFGS-B without bounds can go outside. We should enforce bounds or project.
        # Simple projection:
        for i in range(len(best_x)):
            if norms[i] > 1.0:
                best_x[i] = best_x[i] / norms[i] * 0.9999
                
    return best_x

# EVOLVE-BLOCK-END
