"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 135.0  # safety margin before 150s kill


def _score(pts):
    """Compute the packing score: min of pairwise distances/2 and container margin."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Container margin
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_margin = 1.0 - np.max(radii)
    return min(min_pair, min_margin)


def _init_ring(n_circles):
    """Initialize circles on concentric rings for a good starting configuration."""
    # Use a known good structure: center + rings
    # For 50 circles, use rings of sizes: 1, 6, 12, 18, 13 (or similar)
    # Let's use a hexagonal-like ring structure
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.22, 0.42, 0.62, 0.82]
    
    pts = []
    for size, rad in zip(ring_sizes, ring_radii):
        if size == 1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(size) / size
            pts.extend(np.column_stack([rad * np.cos(angles), rad * np.sin(angles)]))
    
    pts = np.array(pts[:n_circles])
    return pts


def _soft_min_grad(pts, r_target, beta=20.0):
    """
    Compute gradient of a smooth approximation to the max-min objective.
    We want to maximize r such that all constraints are satisfied.
    Use a log-sum-exp style soft-min for the constraints.
    
    Returns: (objective_value, gradient) where gradient points in direction to increase r.
    """
    n = len(pts)
    
    # Compute all pairwise distances and container margins
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    margins = 1.0 - radii
    
    # Soft min of all constraints
    # Constraints: dists[i,j]/2 for i<j, and margins[i]
    # We want to maximize the minimum of these
    
    # For pairwise: use soft-min over all i<j
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_dists, margins])
    
    # Soft min using log-sum-exp trick: soft_min(x) = -logsumexp(-x)/beta
    # Actually, for maximization of min, we can use:
    # f = -log(sum(exp(-beta * x_i))) / beta  (this is a smooth approximation to min)
    # Gradient: df/dx_j = exp(-beta*x_j) / sum(exp(-beta*x_i))
    
    neg_beta_x = -beta * all_constraints
    # Clip for numerical stability
    neg_beta_x = np.clip(neg_beta_x, -500, 500)
    
    log_sum = np.log(np.sum(np.exp(neg_beta_x - np.max(neg_beta_x)))) + np.max(neg_beta_x)
    soft_min_val = -log_sum / beta
    
    # Weights for each constraint
    weights = np.exp(neg_beta_x - log_sum)  # these sum to 1
    
    n_pairs = len(pair_dists)
    pair_weights = weights[:n_pairs]
    margin_weights = weights[n_pairs:]
    
    # Compute gradient with respect to pts
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    # For each pair (i,j), the distance d_ij = |p_i - p_j|
    # d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
    # d(d_ij/2)/d(p_i) = (p_i - p_j) / (2 * d_ij)
    # The soft-min gradient contribution: sum_j w_j * d(constraint_j)/d(p_i)
    
    # We need to map pair_weights back to pairs
    # iu[0] gives i indices, iu[1] gives j indices
    i_idx = iu[0]
    j_idx = iu[1]
    
    # For each pair, contribution to grad[i] and grad[j]
    # d(pair_dist)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(pair_dist)/d(p_j) = (p_j - p_i) / (2 * |p_j - p_i|)
    
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        diff_ij = pts[i] - pts[j]
        dist_ij = np.sqrt(np.sum(diff_ij ** 2))
        if dist_ij < 1e-12:
            continue
        # d(d_ij/2)/d(p_i) = (p_i - p_j) / (2 * d_ij)
        grad[i] += w * diff_ij / (2.0 * dist_ij)
        grad[j] -= w * diff_ij / (2.0 * dist_ij)
    
    # Margin contribution
    # margin_i = 1 - |p_i|
    # d(margin_i)/d(p_i) = -p_i / |p_i|
    for i in range(n):
        w = margin_weights[i]
        if w < 1e-10:
            continue
        r_i = radii[i]
        if r_i < 1e-12:
            continue
        grad[i] -= w * pts[i] / r_i
    
    return soft_min_val, grad


def _project_to_disk(pts):
    """Project points to be inside the unit disk."""
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] /= radii[mask, np.newaxis]
    return pts


def construct():
    start_time = time.time()
    
    # Initialize with a good ring configuration
    pts = _init_ring(N)
    
    # Ensure initial validity
    pts = _project_to_disk(pts)
    
    # Gradient ascent with adaptive step size
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Parameters
    lr = 0.01
    beta = 10.0  # soft-min sharpness
    max_iter = 5000
    
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE:
            break
        
        # Compute soft-min objective and gradient
        obj, grad = _soft_min_grad(pts, 0.0, beta=beta)
        
        # Take gradient step
        pts = pts + lr * grad
        
        # Project to disk
        pts = _project_to_disk(pts)
        
        # Compute actual score
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Adaptive learning rate: reduce if score decreased
        if it > 0:
            prev_score = _score(best_pts)
            if score < prev_score - 1e-6:
                lr *= 0.9
            elif score > prev_score + 1e-6:
                lr = min(lr * 1.05, 0.05)
        
        # Increase beta over time for sharper min approximation
        if it % 200 == 0 and it > 0:
            beta = min(beta * 1.2, 50.0)
        
        # Occasionally try a small random perturbation to escape local minima
        if it % 500 == 0 and it > 0:
            perturbation = np.random.randn(N, 2) * 0.01
            trial_pts = _project_to_disk(pts + perturbation)
            trial_score = _score(trial_pts)
            if trial_score > best_score:
                best_score = trial_score
                best_pts = trial_pts.copy()
                pts = trial_pts.copy()
    
    # Final polish: try to improve with very small steps
    if time.time() - start_time < _DEADLINE - 10:
        lr_small = 0.001
        beta_final = 50.0
        for it in range(2000):
            elapsed = time.time() - start_time
            if elapsed > _DEADLINE - 5:
                break
            obj, grad = _soft_min_grad(best_pts, 0.0, beta=beta_final)
            trial = _project_to_disk(best_pts + lr_small * grad)
            trial_score = _score(trial)
            if trial_score > best_score:
                best_score = trial_score
                best_pts = trial
            else:
                lr_small *= 0.95
    
    # Final validation and cleanup
    # Ensure no points are outside the disk
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] /= radii[mask, np.newaxis]
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        # Fallback to initial
        best_pts = _init_ring(N)
        best_pts = _project_to_disk(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
