"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_soft_min(points, beta):
    """Compute soft-min of pairwise distances/2 and boundary distances.
    
    Uses log-sum-exp for numerical stability.
    Returns (soft_min_value, gradient) where gradient is w.r.t. points.
    """
    n = len(points)
    
    # Boundary distances: 1 - |p_i|
    norms = np.sqrt(np.sum(points**2, axis=1))
    boundary_dists = 1.0 - norms
    
    # Pairwise distances / 2
    # Compute all pairwise differences efficiently
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    # Set diagonal to large value so it doesn't affect min
    np.fill_diagonal(pair_dists, 1e10)
    
    # Collect all distances
    # We want soft min of: all boundary_dists and all pair_dists (upper triangle)
    # For efficiency, work with the full set but use symmetry
    
    # Soft min via log-sum-exp: softmin(x) = -logsumexp(-beta * x) / beta
    # For numerical stability, use the shift trick
    
    # Combine all distances into one array
    # boundary_dists: N values
    # pair_dists: N*(N-1)/2 values (upper triangle)
    # Total: N + N*(N-1)/2
    
    # For N=300, that's 300 + 44850 = 45150 values
    # This is manageable
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_dists[iu]
    
    all_dists = np.concatenate([boundary_dists, pair_vals])
    
    # Soft min computation
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    # = -1/beta * (max(-beta*x) + log(sum(exp(-beta*x - max(-beta*x)))))
    
    neg_beta_x = -beta * all_dists
    max_neg = np.max(neg_beta_x)
    
    # For stability
    exp_terms = np.exp(neg_beta_x - max_neg)
    log_sum = max_neg + np.log(np.sum(exp_terms))
    soft_min_val = -log_sum / beta
    
    # Gradient computation
    # d(softmin)/d(x_k) = -1/beta * d/d(x_k) log(sum(exp(-beta * x_j)))
    # = -1/beta * sum_j(exp(-beta*x_j) * (-beta) * delta(j,k)) / sum_j(exp(-beta*x_j))
    # = sum_j(exp(-beta*x_j) * delta(j,k)) / sum_j(exp(-beta*x_j))
    # = exp(-beta*x_k) / sum_j(exp(-beta*x_j))
    
    # Using shifted version:
    # weights_k = exp(-beta*x_k - max_neg) / sum(exp(-beta*x_j - max_neg))
    
    weights = exp_terms / np.sum(exp_terms)
    
    # Now compute gradient w.r.t. each point
    # boundary_dists[i] = 1 - |p_i|
    # d(boundary_dists[i])/dp_i = -p_i / |p_i|
    
    # pair_dists[j] = |p_a - p_b| / 2 for pair (a,b)
    # d(pair_dists[j])/dp_a = (p_a - p_b) / (2 * |p_a - p_b|) = (p_a - p_b) / (4 * pair_dists[j])
    # d(pair_dists[j])/dp_b = (p_b - p_a) / (2 * |p_b - p_a|) = (p_b - p_a) / (4 * pair_dists[j])
    
    grad = np.zeros_like(points)
    
    # Boundary contribution
    # For each point i, weight w_i contributes to grad[i]
    boundary_weights = weights[:n]
    
    # d(boundary_dist_i)/dp_i = -p_i / |p_i|
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = -points / safe_norms[:, np.newaxis]
    
    grad += boundary_weights[:, np.newaxis] * boundary_grad
    
    # Pairwise contribution
    # For each pair (a, b), weight w contributes:
    # grad[a] += w * (p_a - p_b) / (2 * |p_a - p_b|)
    # grad[b] += w * (p_b - p_a) / (2 * |p_b - p_a|)
    
    # iu gives indices for upper triangle
    ia, ib = iu
    pair_weights = weights[n:]
    
    # Compute unit vectors
    pair_diff = points[ia] - points[ib]  # shape (M, 2)
    pair_norms = np.sqrt(np.sum(pair_diff**2, axis=1))
    safe_pair_norms = np.where(pair_norms > 1e-10, pair_norms, 1.0)
    unit_vecs = pair_diff / safe_pair_norms[:, np.newaxis]
    
    # grad[ia] += pair_weights * unit_vecs
    # grad[ib] -= pair_weights * unit_vecs
    np.add.at(grad, ia, pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, ib, -pair_weights[:, np.newaxis] * unit_vecs)
    
    return soft_min_val, grad

def _optimize_gradient(points, time_limit, beta_start=50.0, beta_end=200.0, lr_start=0.001, lr_end=0.0001):
    """Gradient-based optimization with adaptive step size and increasing beta.
    
    Maximizes the soft-min surrogate, which approximates the true min as beta -> inf.
    """
    best_points = points.copy()
    best_score = None
    start_time = time.time()
    
    # Compute initial score
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    dist_boundary = 1.0 - norms
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(pair_dists, 1e10)
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(pair_dists[iu])
    best_score = min(np.min(dist_boundary), min_pair)
    
    n_iter = 0
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < time_limit:
        # Adaptive beta: increase over time
        progress = min(1.0, n_iter / 3000.0)
        beta = beta_start + (beta_end - beta_start) * progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute soft min and gradient
        soft_val, grad = _compute_soft_min(best_points, beta)
        
        # Take gradient step
        new_points = best_points + lr * grad
        
        # Project back inside unit circle if needed
        new_norms = np.sqrt(np.sum(new_points**2, axis=1))
        mask = new_norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / new_norms[mask, np.newaxis]
        
        # Check if improvement
        new_norms = np.sqrt(np.sum(new_points**2, axis=1))
        new_dist_boundary = 1.0 - new_norms
        diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        new_pair_dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
        np.fill_diagonal(new_pair_dists, 1e10)
        new_min_pair = np.min(new_pair_dists[iu])
        new_score = min(np.min(new_dist_boundary), new_min_pair)
        
        if new_score > best_score + 1e-12:
            best_points = new_points
            best_score = new_score
            patience = 0
        else:
            patience += 1
            # Reduce learning rate if stuck
            lr *= 0.5
            if patience > max_patience:
                break
        
        n_iter += 1
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial hexagonal lattice
    spacing = 0.065
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.97
    
    # Phase 1: Coarse gradient optimization with low beta
    if time.time() < deadline - 80:
        points, score = _optimize_gradient(points, time_limit=30.0, beta_start=20.0, beta_end=80.0, lr_start=0.002, lr_end=0.0005)
    
    # Phase 2: Fine gradient optimization with higher beta
    if time.time() < deadline - 30:
        points, score = _optimize_gradient(points, time_limit=40.0, beta_start=80.0, beta_end=300.0, lr_start=0.0008, lr_end=0.0001)
    
    # Phase 3: Final polish with very high beta and tiny steps
    if time.time() < deadline - 5:
        points, score = _optimize_gradient(points, time_limit=15.0, beta_start=300.0, beta_end=1000.0, lr_start=0.0002, lr_end=0.00002)
    
    return points
# EVOLVE-BLOCK-END
