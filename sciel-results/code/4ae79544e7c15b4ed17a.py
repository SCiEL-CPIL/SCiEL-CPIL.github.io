"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
SEED_RADIUS = 0.1258254895304

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the max common radius for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check if all centers are inside the unit circle
    dists_to_origin = np.linalg.norm(centers, axis=1)
    if np.any(dists_to_origin > 1.0):
        return 0.0
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - dists_to_origin)
    return min(min_pair_dist, min_edge_dist) / 2.0


def _soft_min_obj_and_grad(centers, r, beta):
    """
    Compute the negative soft-min objective and its gradient.
    We want to maximize r, so we minimize -r.
    But we use a smooth surrogate: we want to maximize the min of (d_ij - 2r) and (1 - |c_i| - r).
    Equivalently, minimize the negative soft-min of these terms.
    
    Let's define for each constraint:
      g_ij = d_ij - 2r  (for i<j)
      g_i  = 1 - |c_i| - r
    
    We want to maximize min(g). 
    Soft-max of g approximates min(g) for large beta.
    So we minimize -soft_max(g).
    
    soft_max(g) = (1/beta) * log(sum(exp(beta * g)))
    -soft_max(g) = -(1/beta) * log(sum(exp(beta * g)))
    
    Gradient of -soft_max(g) w.r.t. centers:
    d/dc_i [-soft_max] = - (1/beta) * (1/sum(exp(beta*g))) * sum_j beta * (d g_j / d c_i) * exp(beta * g_j)
    = - sum_j (exp(beta * g_j) / sum_k exp(beta * g_k)) * (d g_j / d c_i)
    
    For g_ij = |c_i - c_j| - 2r:
      d g_ij / d c_i = (c_i - c_j) / |c_i - c_j|
      d g_ij / d c_j = (c_j - c_i) / |c_i - c_j|
    
    For g_i = 1 - |c_i| - r:
      d g_i / d c_i = -c_i / |c_i|  (if |c_i| > 0)
    
    Note: r is treated as a constant here for the gradient w.r.t. centers.
    """
    n = len(centers)
    # Compute all g values
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # g_ij for i<j
    i_idx, j_idx = np.triu_indices(n, k=1)
    g_pair = dists[i_idx, j_idx] - 2 * r
    
    # g_i for edge
    dists_to_origin = np.linalg.norm(centers, axis=1)
    g_edge = 1.0 - dists_to_origin - r
    
    # Combine all g
    g = np.concatenate([g_pair, g_edge])
    
    # Soft-max
    # For numerical stability, subtract max
    g_max = np.max(g)
    exp_g = np.exp(beta * (g - g_max))
    sum_exp = np.sum(exp_g)
    soft_max_val = g_max + (1.0 / beta) * np.log(sum_exp)
    
    obj = -soft_max_val
    
    # Weights
    w = exp_g / sum_exp
    
    # Gradient w.r.t. centers
    grad = np.zeros_like(centers)
    
    # Pairwise part
    # w_pair = w[:len(g_pair)]
    # For each pair (i, j), contribution to grad[i] is -w_ij * (c_i - c_j)/d_ij
    # and to grad[j] is -w_ij * (c_j - c_i)/d_ij
    w_pair = w[:len(g_pair)]
    d_pair = dists[i_idx, j_idx]
    # Avoid division by zero
    d_pair_safe = np.where(d_pair > 1e-12, d_pair, 1.0)
    dir_pair = (centers[i_idx] - centers[j_idx]) / d_pair_safe[:, np.newaxis]
    
    # grad[i] -= w * dir
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_pair)
    np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_pair)
    
    # Edge part
    w_edge = w[len(g_pair):]
    d_edge = dists_to_origin
    d_edge_safe = np.where(d_edge > 1e-12, d_edge, 1.0)
    dir_edge = centers / d_edge_safe[:, np.newaxis]
    # grad[i] -= w_i * (-c_i / |c_i|) = w_i * c_i / |c_i|
    # Wait: d g_i / d c_i = -c_i / |c_i|
    # d/dc_i [-soft_max] = - sum w_k * d g_k / d c_i
    # For edge term k=i: - w_i * (-c_i / |c_i|) = w_i * c_i / |c_i|
    grad += w_edge[:, np.newaxis] * dir_edge
    
    return obj, grad


def _ring_init(n_points):
    """Initialize points on a ring or in a pattern."""
    # Try a few different initializations
    inits = []
    
    # 1. Random uniform in disk
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal lattice approximation
    # Place points in a hex grid
    spacing = 0.2
    pts_hex = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 1:
                pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    if len(pts_hex) >= n_points:
        # Sample n_points from the hex points
        idx = np.random.choice(len(pts_hex), n_points, replace=False)
        inits.append(pts_hex[idx])
    
    # 3. Two rings
    if n_points >= 10:
        n_inner = n_points // 2
        n_outer = n_points - n_inner
        r_inner = 0.3
        r_outer = 0.8
        ang_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        ang_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        pts_rings = np.column_stack([
            np.concatenate([r_inner*np.cos(ang_in), r_outer*np.cos(ang_out)]),
            np.concatenate([r_inner*np.sin(ang_in), r_outer*np.sin(ang_out)])
        ])
        inits.append(pts_rings)
    
    return inits


def _gradient_ascent(centers, r, beta, max_iter=1000, lr=0.01, deadline=None):
    """
    Gradient ascent on the soft-min objective to maximize r.
    """
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        obj, grad = _soft_min_obj_and_grad(centers, r, beta)
        
        # Update centers
        centers = centers + lr * grad
        
        # Project onto unit disk
        dists = np.linalg.norm(centers, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            centers[mask] = centers[mask] / dists[mask, np.newaxis]
        
        # Occasionally recompute score and update best
        if it % 50 == 0:
            score = _compute_score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                # Increase r slightly if we made progress
                # This is a heuristic
                pass
        
        # Adaptive learning rate decay
        lr *= 0.999
    
    return best_centers, best_score


def _refine_with_soft_min(centers, r, beta, deadline):
    """Refine a given configuration using soft-min gradient ascent."""
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Try different learning rates and betas
    for lr in [0.001, 0.005, 0.01, 0.02]:
        if time.time() > deadline:
            break
        c, s = _gradient_ascent(centers, r, beta, max_iter=500, lr=lr, deadline=deadline)
        if s > best_score:
            best_score = s
            best_centers = c
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, with safety margin
    
    # Initial radius estimate
    r_current = 0.12
    
    # Try multiple initializations
    best_centers = None
    best_score = 0.0
    
    # Seed with a known good configuration if possible
    # For now, use random inits
    
    n_inits = 20
    for init_idx in range(n_inits):
        if time.time() > deadline - 10:  # Leave 10s for final polish
            break
        
        inits = _ring_init(N)
        # Add some randomization
        for base_init in inits:
            if time.time() > deadline - 10:
                break
            
            centers = base_init + np.random.normal(0, 0.01, (N, 2))
            
            # Initial score
            score = _compute_score(centers)
            
            # Refine
            centers_ref, score_ref = _refine_with_soft_min(centers, r_current, beta=100, deadline=deadline)
            
            if score_ref > best_score:
                best_score = score_ref
                best_centers = centers_ref
                r_current = score_ref
    
    # Final polish: try to increase r further with higher beta
    if best_centers is not None and time.time() < deadline - 5:
        for beta in [50, 100, 200, 500]:
            if time.time() > deadline - 5:
                break
            c, s = _refine_with_soft_min(best_centers, r_current, beta=beta, deadline=deadline)
            if s > best_score:
                best_score = s
                best_centers = c
                r_current = s
    
    if best_centers is None:
        # Fallback: return a valid configuration
        # Use the hexagonal lattice from the seed
        spacing = 0.2
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= 1:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            return points[:N]
        else:
            # Just return random points
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.9, N))
            return np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    
    return best_centers

# EVOLVE-BLOCK-END
