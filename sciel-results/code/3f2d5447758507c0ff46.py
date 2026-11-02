"""
Packing 300 equal circles in a unit circle.
Uses a smooth surrogate (soft-min) for the objective function and L-BFGS-B for optimization.
Includes a time guard to ensure completion within 150s.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5 seconds safety margin

def compute_radius(C):
    """
    Computes the maximum radius for the given centers C.
    Returns (radius, is_valid).
    """
    if C.shape != (N, 2):
        return 0.0, False
    if not np.all(np.isfinite(C)):
        return 0.0, False
        
    # Pairwise distances
    # C is N x 2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.sqrt(np.sum(C**2, axis=1))
    min_boundary_dist = np.min(1.0 - norms)
    
    radius = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Validity check
    if radius <= 0 or min_boundary_dist < 0:
        return 0.0, False
        
    return radius, True

def softmin(v, alpha):
    """
    Numerically stable soft-min of vector v.
    softmin(v) ~= min(v) as alpha -> inf
    """
    v_max = np.max(v)
    # Subtract max to prevent overflow
    exp_terms = np.exp(-alpha * (v - v_max))
    sum_exp = np.sum(exp_terms)
    return v_max - np.log(sum_exp) / alpha

def objective_wrapper(C, alpha, penalty_weight):
    """
    Objective function for L-BFGS-B.
    Minimizes -softmin(distances) + penalty.
    """
    C = C.reshape(-1, 2)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists.flatten() / 2.0
    
    # Boundary distances
    norms = np.sqrt(np.sum(C**2, axis=1))
    boundary_dists = 1.0 - norms
    
    # Combine distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Softmin
    score = -softmin(all_dists, alpha)
    
    # Penalty for points outside the unit circle
    outside = np.maximum(0.0, norms - 1.0)
    penalty = penalty_weight * np.sum(outside**2)
    
    return score + penalty

def gradient_wrapper(C, alpha, penalty_weight):
    """
    Analytical gradient of the objective function.
    """
    C = C.reshape(-1, 2)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0 # N x N matrix
    
    # Boundary distances
    norms = np.sqrt(np.sum(C**2, axis=1))
    boundary_dists = 1.0 - norms # N vector
    
    # We need the weights for the softmin
    all_dists_flat = np.concatenate([pair_dists.flatten(), boundary_dists])
    v_max = np.max(all_dists_flat)
    exp_terms = np.exp(-alpha * (all_dists_flat - v_max))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Split weights back
    num_pairs = N * (N - 1)
    w_pairs = weights[:num_pairs].reshape(N, N)
    w_boundary = weights[num_pairs:]
    
    # Gradient of softmin with respect to C
    # d(softmin)/dC_i = - sum_j w_ij * d(pair_dist_ij)/dC_i - w_i * d(boundary_dist_i)/dC_i
    # Objective is -softmin, so gradient is + sum ...
    
    grad = np.zeros_like(C)
    
    # Pair contributions
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    coeff = w_pairs / safe_dists
    grad_pairs = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary contributions
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -w_boundary[:, np.newaxis] * C / safe_norms[:, np.newaxis]
    
    # Penalty gradient
    outside_mask = norms > 1.0
    grad_penalty = np.zeros_like(C)
    if np.any(outside_mask):
        grad_penalty[outside_mask] = penalty_weight * 2.0 * (norms[outside_mask] - 1.0)[:, np.newaxis] * C[outside_mask] / safe_norms[outside_mask][:, np.newaxis]
        
    total_grad = grad_pairs + grad_boundary + grad_penalty
    
    return total_grad.flatten()

def construct():
    global START_TIME, TIME_LIMIT
    
    # 1. Initialization
    s = 0.11
    k = int(1.0 / s) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - s/2.0:
                rows.append([x, y])
                
    points = np.array(rows)
    
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        missing = N - len(points)
        random_pts = np.random.rand(missing, 2) * 2.0 - 1.0
        norms = np.sqrt(np.sum(random_pts**2, axis=1))
        random_pts *= (0.9 / np.max(norms))[:, np.newaxis]
        points = np.vstack([points, random_pts])
        
    # Add jitter
    points += np.random.normal(0, 0.01, points.shape)
    
    best_C = points.copy()
    best_score = compute_radius(best_C)[0]
    
    # 2. Optimization Loop
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    penalty_weight = 100.0
    
    for alpha in alphas:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        try:
            res = minimize(
                objective_wrapper,
                points.flatten(),
                jac=gradient_wrapper,
                method='L-BFGS-B',
                args=(alpha, penalty_weight),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_C = res.x.reshape(-1, 2)
            new_score, valid = compute_radius(new_C)
            
            if valid and new_score > best_score:
                best_score = new_score
                best_C = new_C
                points = new_C.copy()
                
        except Exception:
            pass
            
        if time.time() - START_TIME > TIME_LIMIT:
            break

    if not np.all(np.isfinite(best_C)):
         return points[np.argsort(np.hypot(points[:,0], points[:,1]), kind='stable')[:N]]
         
    return best_C

# EVOLVE-BLOCK-END
