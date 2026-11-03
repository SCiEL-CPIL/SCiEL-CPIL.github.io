"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the packing radius: min of half-min-distance and min(1-|c_i|)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    return min(min_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """
    Smooth approximation to min(values) using log-sum-exp.
    As beta -> infinity, soft_min -> true min.
    """
    # Use the log-sum-exp trick for numerical stability
    m = np.min(values)
    # If all values are very close, just return m
    if np.max(values) - m < 1e-12:
        return m
    # soft_min = -log(sum(exp(-beta * (v - m)))) / beta + m
    # = m - (1/beta) * log(sum(exp(-beta * (v - m))))
    exps = np.exp(-beta * (values - m))
    log_sum = np.log(np.sum(exps))
    return m - log_sum / beta

def _soft_min_grad(values, beta):
    """
    Gradient of soft_min with respect to values.
    Returns array of same shape as values.
    """
    m = np.min(values)
    if np.max(values) - m < 1e-12:
        return np.ones_like(values) / len(values)
    exps = np.exp(-beta * (values - m))
    sum_exps = np.sum(exps)
    # d(soft_min)/d(v_i) = -exp(-beta*(v_i - m)) / sum_exps
    return -exps / sum_exps

def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        pass

    # Sort by distance to center and take the first n
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]

def _compute_soft_min_objective_and_grad(pts, beta):
    """
    Compute the soft-min objective (to maximize) and its gradient.
    Objective = soft_min([dist_ij/2 for all i<j, (1-|c_i|) for all i])
    """
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    # All pairwise distances (upper triangle)
    iu, ju = np.triu_indices(n, k=1)
    pair_dists = dist[iu, ju]
    
    # Boundary gaps
    r = np.hypot(pts[:, 0], pts[:, 1])
    boundary_gaps = 1.0 - r
    
    # Combine all constraints: pair_dists/2 and boundary_gaps
    all_values = np.concatenate([pair_dists / 2.0, boundary_gaps])
    
    # Soft min value
    soft_min_val = _soft_min(all_values, beta)
    
    # Gradient of soft min w.r.t. all_values
    val_grad = _soft_min_grad(all_values, beta)
    
    # Now compute gradient w.r.t. pts
    grad_pts = np.zeros_like(pts)
    
    # Pairwise distances: d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(soft_min)/d(c_i) = sum_j val_grad[j] * d(pair_dist_j / 2) / d(c_i)
    for idx in range(len(iu)):
        i, j = iu[idx], ju[idx]
        d = dist[i, j]
        if d < 1e-10:
            continue
        # val_grad[idx] is the weight for this pair
        vec = (pts[i] - pts[j]) / d
        grad_pts[i] += val_grad[idx] * vec * 0.5  # * 0.5 because of /2
        grad_pts[j] -= val_grad[idx] * vec * 0.5
    
    # Boundary: d(boundary_gap_i)/d(c_i) = -c_i / |c_i|
    for i in range(n):
        ri = r[i]
        if ri < 1e-10:
            continue
        vec = -pts[i] / ri
        grad_pts[i] += val_grad[len(pair_dists) + i] * vec
    
    return soft_min_val, grad_pts

def _optimize_soft_min(pts, time_limit, beta_start=100.0, beta_end=10000.0):
    """
    Optimize packing using smooth soft-min surrogate with gradient ascent.
    Two-phase: coarse (low beta) then fine (high beta) polish.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Phase 1: Coarse optimization with moderate beta
    # Phase 2: Fine polish with very high beta
    
    phases = [
        (beta_start, 40.0, 0.008),   # beta, time_fraction, step_size
        (beta_start * 10, 35.0, 0.004),
        (beta_end, 30.0, 0.002),
    ]
    
    for beta, time_frac, base_step in phases:
        phase_time = time_limit * time_frac / 100.0
        phase_end = start_time + phase_time
        
        step_size = base_step
        step_decay = 0.9995
        
        it = 0
        while time.time() < phase_end:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 3.0:
                break
            
            # Compute soft-min objective and gradient
            soft_val, grad = _compute_soft_min_objective_and_grad(pts, beta)
            
            # Compute true score for tracking
            true_score = _compute_score(pts)
            if true_score > best_score:
                best_score = true_score
                best_pts = pts.copy()
            
            # Gradient ascent
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            grad /= grad_norm
            pts_new = pts + step_size * grad
            
            # Project inside unit circle
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask_out = r_new >= 1.0
            if np.any(mask_out):
                pts_new[mask_out] *= (0.999 / r_new[mask_out])
            
            # Accept if it improves the soft-min objective (proxy for true score)
            soft_val_new, _ = _compute_soft_min_objective_and_grad(pts_new, beta)
            
            if soft_val_new > soft_val:
                pts = pts_new
            else:
                # Small perturbation to escape local minima
                if it % 20 == 0:
                    noise = np.random.normal(0, 0.003, size=pts.shape)
                    pts_pert = pts + noise
                    r_p = np.hypot(pts_pert[:, 0], pts_pert[:, 1])
                    if np.all(r_p < 1.0):
                        score_p = _compute_score(pts_pert)
                        if score_p > best_score:
                            pts = pts_pert
                        else:
                            # Try smaller step
                            step_size *= 0.9
            
            step_size *= step_decay
            if step_size < 1e-7:
                step_size = base_step * 0.5
            
            it += 1
    
    return best_pts, best_score

def _polish(pts, time_limit):
    """
    Final high-precision polish using very sharp soft-min.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    beta = 50000.0
    step_size = 0.001
    step_decay = 0.9999
    
    it = 0
    while time.time() < start_time + time_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2.0:
            break
        
        soft_val, grad = _compute_soft_min_objective_and_grad(pts, beta)
        
        true_score = _compute_score(pts)
        if true_score > best_score:
            best_score = true_score
            best_pts = pts.copy()
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        grad /= grad_norm
        pts_new = pts + step_size * grad
        
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask_out = r_new >= 1.0
        if np.any(mask_out):
            pts_new[mask_out] *= (0.999 / r_new[mask_out])
        
        soft_val_new, _ = _compute_soft_min_objective_and_grad(pts_new, beta)
        
        if soft_val_new > soft_val:
            pts = pts_new
        else:
            step_size *= 0.95
        
        step_size *= step_decay
        it += 1
    
    return best_pts, best_score

def construct():
    """
    Construct N=300 equal circles in unit disk.
    Uses hexagonal lattice init + two-phase soft-min optimization + polish.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate initial hexagonal lattice
    pts = _generate_hex_init(N)
    
    # Main optimization with soft-min (two-phase)
    main_pts, main_score = _optimize_soft_min(pts, total_time_limit * 0.85)
    
    # Final polish with very sharp soft-min
    final_pts, final_score = _polish(main_pts, total_time_limit * 0.15)
    
    # Final validation: ensure all points are strictly inside
    r = np.hypot(final_pts[:, 0], final_pts[:, 1])
    for i in range(len(final_pts)):
        if r[i] >= 1.0:
            final_pts[i] *= 0.999 / r[i]
    
    return final_pts
# EVOLVE-BLOCK-END
