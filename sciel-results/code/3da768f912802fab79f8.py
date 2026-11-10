import numpy as np
import time

N = 50

def _calc_min_radius(points):
    # min distance between pairs
    # min distance to boundary
    # points is (N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _compute_gradient(points, r_target):
    """Compute gradient direction to increase the minimum radius.
    
    Uses a weighted sum of all near-binding constraints, not just the single
    most binding one, to get a more robust gradient direction.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    d_min = np.min(dists)
    b_min = np.min(boundary_dists)
    
    r = min(d_min / 2.0, b_min)
    
    grad = np.zeros_like(points)
    
    # Use a wider tolerance to include near-binding constraints
    # This gives a more robust gradient direction
    tol = max(1e-4, 0.02 * d_min)
    
    # Pair constraints: weight by how close to binding
    # Use exponential weighting for smooth transition
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_dists = dists[i_idx, j_idx]
    
    # Weight: higher for closer pairs, smooth falloff
    pair_weights = np.exp(-pair_dists / (0.1 * d_min + 1e-10))
    pair_weights[pair_dists > d_min + tol] = 0.0
    
    for idx in range(len(pair_dists)):
        i = i_idx[idx]
        j = j_idx[idx]
        d = pair_dists[idx]
        if d < 1e-10:
            continue
        w = pair_weights[idx]
        if w < 1e-10:
            continue
        v = (points[i] - points[j]) / d
        grad[i] += w * v
        grad[j] -= w * v
    
    # Boundary constraints: weight by how close to binding
    b_weights = np.exp(-boundary_dists / (0.1 * b_min + 1e-10))
    b_weights[boundary_dists > b_min + tol] = 0.0
    
    for i in range(N):
        r_i = radii[i]
        if r_i < 1e-10:
            continue
        w = b_weights[i]
        if w < 1e-10:
            continue
        v = -points[i] / r_i
        grad[i] += w * v
        
    # Normalize gradient to unit length for stable stepping
    norm = np.linalg.norm(grad)
    if norm > 1e-10:
        grad /= norm
        
    return grad, r

def _soft_min_energy(points, r, beta):
    """Compute soft-min energy and its gradient using log-sum-exp.
    
    Uses slacks: s_ij = d_ij - 2r, t_i = (1-r) - |c_i|
    soft_min = -log(sum exp(-beta * s)) / beta approximates min slack.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Pair slacks: s_ij = d_ij - 2r
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_slacks = dists[i_idx, j_idx] - 2.0 * r
    
    # Boundary slacks: t_i = (1-r) - |c_i|
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_slacks = (1.0 - r) - radii
    
    # Combine all slacks
    all_slacks = np.concatenate([pair_slacks, boundary_slacks])
    
    # Soft min using log-sum-exp (numerically stable)
    m = np.min(all_slacks)
    shifted = all_slacks - m  # now min is 0
    exp_vals = np.exp(-beta * shifted)
    log_sum_exp = m + np.log(np.sum(exp_vals))
    soft_min_val = -log_sum_exp / beta
    
    # Weights for each constraint
    weights = exp_vals / np.sum(exp_vals)
    pair_weights = weights[:len(pair_slacks)]
    boundary_weights = weights[len(pair_slacks):]
    
    grad = np.zeros_like(points)
    
    # Pair contributions: d(s_ij)/d(c_i) = (c_i - c_j) / d_ij
    for idx in range(len(pair_slacks)):
        i = i_idx[idx]
        j = j_idx[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = pair_weights[idx]
        if w < 1e-12:
            continue
        v = (points[i] - points[j]) / d
        grad[i] += w * v
        grad[j] -= w * v
    
    # Boundary contributions: d(t_i)/d(c_i) = -c_i / |c_i|
    for idx in range(N):
        r_i = radii[idx]
        if r_i < 1e-10:
            continue
        w = boundary_weights[idx]
        if w < 1e-12:
            continue
        v = -points[idx] / r_i
        grad[idx] += w * v
    
    return soft_min_val, grad

def _smooth_optimize(points, start_r, deadline, max_iters=10000):
    """Use smooth soft-min energy to optimize the packing with adaptive beta."""
    beta = 20.0  # Start with moderate sharpness
    step_size = 5e-4
    best_r = start_r
    best_points = points.copy()
    consecutive_failures = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current actual r
        current_r = _calc_min_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            consecutive_failures = 0
        
        # Compute soft-min energy and gradient
        soft_val, grad = _soft_min_energy(points, current_r, beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        grad /= grad_norm  # Normalize
        
        # Try multiple step sizes to find improvement
        improved = False
        for factor in [1.0, 0.5, 0.25, 0.1]:
            trial_step = step_size * factor
            new_points = points + trial_step * grad
            new_r = _calc_min_radius(new_points)
            
            if new_r > current_r + 1e-15:
                points = new_points
                current_r = new_r
                improved = True
                # Adjust step size based on which factor worked
                step_size = trial_step
                break
        
        if improved:
            # Increase beta to make soft-min sharper
            beta = min(beta * 1.005, 500.0)
        else:
            consecutive_failures += 1
            step_size *= 0.7
            if step_size < 1e-9 or consecutive_failures > 50:
                break
    
    return best_points, best_r

def _polish(points, deadline):
    """Final high-precision polish pass with adaptive step size and multi-direction search."""
    best_r = _calc_min_radius(points)
    best_points = points.copy()
    
    step_size = 1e-5
    consecutive_failures = 0
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        grad, current_r = _compute_gradient(points, current_r)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        grad /= grad_norm
        
        # Try multiple step sizes
        improved = False
        for factor in [1.0, 0.5, 0.25]:
            trial_step = step_size * factor
            new_points = points + trial_step * grad
            new_r = _calc_min_radius(new_points)
            
            if new_r > current_r + 1e-16:
                points = new_points
                current_r = new_r
                improved = True
                if current_r > best_r:
                    best_r = current_r
                    best_points = points.copy()
                step_size = trial_step
                break
        
        if improved:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            step_size *= 0.8
            if step_size < 1e-13 or consecutive_failures > 30:
                break
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = 115.0  # 120s limit, keep margin
    
    # 1. Initialization: Hexagonal lattice with center bias
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside the circle
        points, radius = points[radius <= 1.0 - spacing * 0.6], radius[radius <= 1.0 - spacing * 0.6]
        if len(points) >= N:
            # Prefer points closer to center for better packing
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.97
        
    current_r = _calc_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Phase 1: Gradient-based local search with adaptive step
    step_size = 5e-4
    phase1_deadline = start_time + 25.0
    consecutive_failures = 0
    
    while time.time() - start_time < phase1_deadline:
        grad, current_r = _compute_gradient(points, current_r)
        
        if np.linalg.norm(grad) < 1e-10:
            break
            
        # Try multiple step sizes
        improved = False
        for factor in [1.0, 0.5, 0.25]:
            trial_step = step_size * factor
            new_points = points + trial_step * grad
            new_r = _calc_min_radius(new_points)
            
            if new_r > current_r + 1e-15:
                points = new_points
                current_r = new_r
                improved = True
                if current_r > best_r:
                    best_r = current_r
                    best_points = points.copy()
                step_size = trial_step
                break
        
        if improved:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            step_size *= 0.8
            if step_size < 1e-8 or consecutive_failures > 30:
                break
                
    # Phase 2: Smooth soft-min optimization
    phase2_deadline = start_time + 80.0
    if time.time() < phase2_deadline:
        points, current_r = _smooth_optimize(points, current_r, phase2_deadline, max_iters=10000)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
    
    # Phase 3: High-precision polish
    phase3_deadline = deadline - 3.0
    if time.time() < phase3_deadline:
        points, current_r = _polish(best_points, phase3_deadline)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
    
    return best_points
