"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _pair_dist_matrix(p):
    """Compute pairwise distance matrix for N points."""
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return dists


def _compute_radius(p):
    """Compute the common radius for a set of centres inside [-0.5, 0.5]^3."""
    # Boundary constraint: distance from each center to nearest wall
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise distance constraint
    dists = _pair_dist_matrix(p)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)


def _softmin_radius(p, beta):
    """
    Compute a smooth approximation of the minimum clearance using softmin.
    This is differentiable and allows gradient-based optimization.
    """
    # Boundary clearances
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    # Pairwise clearances (half distances)
    dists = _pair_dist_matrix(p)
    np.fill_diagonal(dists, np.inf)
    pair_clearances = dists / 2.0
    
    # Flatten all clearances into a single vector
    all_clearances = np.concatenate([b, pair_clearances[pair_clearances < np.inf]])
    
    # Softmin: approximates min using log-sum-exp with negative sign
    # softmin(x) = -log(sum(exp(-beta * x))) / beta
    # For stability, subtract the min first
    min_c = np.min(all_clearances)
    exp_vals = np.exp(-beta * (all_clearances - min_c))
    softmin_val = min_c - np.log(np.sum(exp_vals)) / beta
    
    return softmin_val


def _smooth_obj(p, beta):
    """
    Smooth objective: maximize the softmin radius.
    We minimize the negative softmin radius.
    """
    return -_softmin_radius(p, beta)


def _smooth_grad(p, beta):
    """
    Compute gradient of the smooth objective w.r.t. positions.
    Uses automatic differentiation via finite differences for simplicity,
    but we can derive analytical gradients for better performance.
    
    For now, use a semi-analytical approach:
    The softmin weight for each clearance is proportional to exp(-beta * c_i).
    The gradient of the softmin w.r.t. p is the weighted sum of gradients of each clearance.
    """
    N_pts = p.shape[0]
    grad = np.zeros_like(p)
    
    # Boundary clearances: b_i = 0.5 - max(|p_i|)
    # Gradient of b_i w.r.t. p_i is -sign(p_i) for the dominant axis
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    b = 0.5 - max_abs
    
    # Pairwise clearances: c_ij = |p_i - p_j| / 2
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_clearances = dists / 2.0
    
    # Build the full clearance vector and compute softmin weights
    all_clearances = np.concatenate([b, pair_clearances[pair_clearances < np.inf]])
    min_c = np.min(all_clearances)
    exp_vals = np.exp(-beta * (all_clearances - min_c))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient contribution from boundary clearances
    # For each point i, the boundary clearance b_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    # The gradient is -sign(p_i[axis]) for the axis that achieves the max
    for i in range(N_pts):
        w_b = weights[i]
        # Find which axis achieves the max
        max_axis = np.argmax(abs_p[i])
        grad[i, max_axis] -= w_b * np.sign(p[i, max_axis])
    
    # Gradient contribution from pairwise clearances
    # For pair (i,j), clearance c_ij = |p_i - p_j| / 2
    # Gradient w.r.t. p_i is (p_i - p_j) / (2 * |p_i - p_j|)
    # Gradient w.r.t. p_j is -(p_i - p_j) / (2 * |p_i - p_j|)
    pair_idx = np.argwhere(pair_clearances < np.inf)
    for (i, j) in pair_idx:
        w_p = weights[N_pts + int(i * (N_pts - 1) + j - (1 if j > i else 0))]
        if i == j:
            continue
        d = diffs[i, j]
        dist = dists[i, j]
        if dist > 1e-12:
            unit = d / dist
            # Gradient of c_ij w.r.t. p_i is unit / 2, w.r.t. p_j is -unit / 2
            grad[i] += w_p * unit / 2.0
            grad[j] -= w_p * unit / 2.0
    
    # The objective is -softmin, so we negate the gradient
    return -grad


def _repulsion_anneal(p, start_beta, end_beta, n_iter, time_limit, start_time):
    """
    Anneal the softmin parameter from start_beta to end_beta.
    Low beta: smooth, many terms contribute (global optimization)
    High beta: sharp, focuses on the minimum (local refinement)
    """
    p = p.copy()
    current_r = _compute_radius(p)
    
    for it in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Interpolate beta
        t = it / max(n_iter - 1, 1)
        beta = start_beta + (end_beta - start_beta) * t
        
        # Compute gradient
        grad = _smooth_grad(p, beta)
        
        # Adaptive step size
        step = 0.005 * (1.0 + 0.5 * np.exp(-beta / 50.0))
        
        # Update
        p_new = p + step * grad
        
        # Clamp to cube
        p_new = np.clip(p_new, -0.4999, 0.4999)
        
        # Check if this improves the actual radius
        r_new = _compute_radius(p_new)
        if r_new > current_r:
            p = p_new
            current_r = r_new
        else:
            # Try a smaller step
            p_new = p + step * 0.1 * grad
            p_new = np.clip(p_new, -0.4999, 0.4999)
            r_new = _compute_radius(p_new)
            if r_new > current_r:
                p = p_new
                current_r = r_new
    
    return p, current_r


def _smooth_polish(p, time_limit, start_time):
    """
    Final polishing phase with high beta to refine the minimum clearance.
    """
    p = p.copy()
    current_r = _compute_radius(p)
    
    beta = 200.0
    n_iter = 500
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad = _smooth_grad(p, beta)
        
        # Very small steps for polishing
        step = 0.0005
        
        p_new = p + step * grad
        p_new = np.clip(p_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(p_new)
        if r_new > current_r:
            p = p_new
            current_r = r_new
    
    return p, current_r


def _local_search(p, time_limit, start_time):
    """
    Local search: try random perturbations and accept if they improve the radius.
    """
    p = p.copy()
    current_r = _compute_radius(p)
    
    n_iter = 200
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        noise = np.random.randn(*p.shape) * 0.005
        p_new = p + noise
        p_new = np.clip(p_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(p_new)
        if r_new > current_r:
            p = p_new
            current_r = r_new
    
    return p, current_r


def _seed_fcc():
    """Generate initial positions using FCC lattice."""
    points = []
    max_idx = 5
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k])
                points.append([i+0.5, j, k+0.5])
                points.append([i, j+0.5, k+0.5])
                
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()
    
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.35 / max_coord
        pts *= scale
        
    return pts


def _seed_grid():
    """Generate initial positions using a regular grid."""
    # 4x4x4 = 64 points, take 50
    n_per_dim = 4
    pts = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                pts.append([(i - (n_per_dim-1)/2) / (n_per_dim-1) * 0.8,
                           (j - (n_per_dim-1)/2) / (n_per_dim-1) * 0.8,
                           (k - (n_per_dim-1)/2) / (n_per_dim-1) * 0.8])
    pts = np.array(pts[:N])
    return pts


def _seed_random():
    """Generate random initial positions."""
    return np.random.uniform(-0.4, 0.4, size=(N, 3))


def _seed_shell_core():
    """Generate positions with some in core, some in shell."""
    # Core: 20 points in a smaller region
    core = np.random.uniform(-0.2, 0.2, size=(20, 3))
    # Shell: 30 points in a larger region
    shell = np.random.uniform(-0.4, 0.4, size=(30, 3))
    pts = np.vstack([core, shell])
    return pts


def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and keep the best
    seeds = [_seed_fcc(), _seed_grid(), _seed_random(), _seed_shell_core()]
    
    best_pts = None
    best_r = 0.0
    
    for seed_pts in seeds:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        # Run repulsion annealing
        p, r = _repulsion_anneal(seed_pts, start_beta=5.0, end_beta=100.0, 
                                   n_iter=300, time_limit=time_limit, start_time=start_time)
        
        # Polish
        p, r = _smooth_polish(p, time_limit=time_limit, start_time=start_time)
        
        # Local search
        p, r = _local_search(p, time_limit=time_limit, start_time=start_time)
        
        if r > best_r:
            best_r = r
            best_pts = p
    
    # If we still have time, do more refinement on the best
    if best_pts is not None and time.time() - start_time < time_limit - 5.0:
        # Additional polishing rounds
        for _ in range(3):
            if time.time() - start_time > time_limit - 5.0:
                break
            best_pts, best_r = _smooth_polish(best_pts, time_limit=time_limit, start_time=start_time)
            best_pts, best_r = _local_search(best_pts, time_limit=time_limit, start_time=start_time)
    
    return best_pts if best_pts is not None else _seed_fcc()
# EVOLVE-BLOCK-END
