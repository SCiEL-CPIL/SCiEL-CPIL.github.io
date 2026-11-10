import numpy as np
import time

N = 50

def _calc_min_radius(points):
    # min distance between pairs
    # min distance to boundary
    # points is (N, 2)
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _compute_gradient(points, r_target):
    # We want to move points to increase the minimum radius.
    # This is equivalent to resolving violations for a slightly larger r.
    # Or simply, compute gradient of the "violation" energy for r = current_r + epsilon.
    
    # Let's use the violation energy gradient approach.
    # Energy E = sum max(0, 2r - d_ij)^2 + sum max(0, |c_i| - (1-r))^2
    # We want to minimize E for a fixed r.
    # If E > 0, we have violations.
    # If E == 0, we are feasible.
    
    # To find the direction to increase r, we can look at the gradient of the active constraints.
    # Let d_min = min d_ij. Let b_min = min (1 - |c_i|).
    # r = min(d_min/2, b_min).
    
    # If d_min/2 < b_min:
    #   The binding constraint is the pair distance.
    #   We want to increase d_min.
    #   Gradient direction is sum of (c_i - c_j)/d_ij for active pairs.
    # If b_min < d_min/2:
    #   The binding constraint is the boundary.
    #   We want to increase b_min (move inwards).
    #   Gradient direction is sum of -c_i/|c_i| for active points.
    # If both:
    #   Combine them.
    
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    d_min = np.min(dists)
    b_min = np.min(boundary_dists)
    
    r = min(d_min / 2.0, b_min)
    
    grad = np.zeros_like(points)
    
    # Threshold for "active"
    tol = 1e-6
    
    if d_min / 2.0 <= b_min + tol:
        # Pair constraints are active or co-active
        # Find all pairs with dist <= d_min + tol
        # This is O(N^2)
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= d_min + tol:
                    d = dists[i, j]
                    if d < 1e-10: continue
                    # Gradient of d_ij wrt c_i is (c_i - c_j) / d
                    # We want to increase d, so move c_i away from c_j
                    v = (points[i] - points[j]) / d
                    grad[i] += v
                    grad[j] -= v
                    
    if b_min <= d_min / 2.0 + tol:
        # Boundary constraints are active
        for i in range(N):
            if boundary_dists[i] <= b_min + tol:
                r_i = radii[i]
                if r_i < 1e-10: continue
                # Gradient of b_i = 1 - |c_i| wrt c_i is -c_i / |c_i|
                # We want to increase b_i, so move in direction -c_i/|c_i|
                v = -points[i] / r_i
                grad[i] += v
                
    # Normalize gradient to unit length for stable stepping
    norm = np.linalg.norm(grad)
    if norm > 1e-10:
        grad /= norm
        
    return grad, r

def _soft_min_energy(points, r, beta):
    """Compute soft-min energy and its gradient using log-sum-exp.
    
    We want to maximize r such that:
      d_ij >= 2r for all i<j
      |c_i| <= 1-r for all i
    
    Define violations:
      v_ij = 2r - d_ij  (positive when violated)
      w_i = |c_i| - (1-r)  (positive when violated)
    
    Soft energy: E = -log(sum exp(-beta * v_ij) + sum exp(-beta * w_i)) / beta
    This approximates min(v_ij, w_i) as beta -> inf.
    
    We want to find direction that increases the minimum, i.e., decreases the max violation.
    Actually, let's use a different approach:
    
    Define the "slack" for each constraint:
      s_ij = d_ij - 2r  (should be >= 0)
      t_i = (1-r) - |c_i|  (should be >= 0)
    
    We want to maximize the minimum slack. Use soft-min of slacks.
    soft_min = -log(sum exp(-beta * s_ij) + sum exp(-beta * t_i)) / beta
    
    Gradient of soft_min wrt points gives direction to increase the minimum slack.
    """
    # Compute all pair distances
    dists_sq = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Pair slacks: s_ij = d_ij - 2r
    # Only consider i < j to avoid double counting
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_slacks = dists[i_idx, j_idx] - 2.0 * r
    
    # Boundary slacks: t_i = (1-r) - |c_i|
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_slacks = (1.0 - r) - radii
    
    # Combine all slacks
    all_slacks = np.concatenate([pair_slacks, boundary_slacks])
    
    # Soft min using log-sum-exp (numerically stable)
    # soft_min = -log(sum exp(-beta * s)) / beta
    # = (log(sum exp(-beta * s)) + beta * max(s)) / (-beta) ... let's be careful
    # Actually: -log(sum exp(-beta*s)) / beta
    # Let m = min(all_slacks) for numerical stability
    m = np.min(all_slacks)
    shifted = all_slacks - m  # now min is 0
    log_sum_exp = m + np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft_min wrt points
    # d(soft_min)/d(point_k) = sum_j [ -exp(-beta * s_j) / (beta * sum exp(-beta * s_j)) * d(s_j)/d(point_k) ]
    # = -sum_j [ exp(-beta * s_j) / sum exp(-beta * s_j) * d(s_j)/d(point_k) ] / beta
    
    # Weights for each constraint
    weights = np.exp(-beta * shifted) / np.sum(np.exp(-beta * shifted))
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
        v = (points[i] - points[j]) / d
        grad[i] += w * v
        grad[j] -= w * v
    
    # Boundary contributions: d(t_i)/d(c_i) = -c_i / |c_i|
    for idx in range(N):
        r_i = radii[idx]
        if r_i < 1e-10:
            continue
        w = boundary_weights[idx]
        v = -points[idx] / r_i
        grad[idx] += w * v
    
    # The gradient of soft_min points in the direction that increases the minimum slack
    # So we should move in the direction of +grad
    return soft_min_val, grad

def _smooth_optimize(points, start_r, deadline, max_iters=5000):
    """Use smooth soft-min energy to optimize the packing."""
    beta = 50.0  # Start with moderate sharpness
    step_size = 1e-4
    best_r = start_r
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current actual r
        current_r = _calc_min_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Compute soft-min energy and gradient
        soft_val, grad = _soft_min_energy(points, current_r, beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        grad /= grad_norm  # Normalize
        
        # Try step
        new_points = points + step_size * grad
        new_r = _calc_min_radius(new_points)
        
        if new_r > current_r + 1e-15:
            points = new_points
            current_r = new_r
            # Increase step size slowly
            step_size = min(step_size * 1.02, 5e-3)
            # Increase beta to make soft-min sharper
            beta = min(beta * 1.001, 200.0)
        else:
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-8:
                break
    
    return best_points, best_r

def _polish(points, deadline):
    """Final high-precision polish pass with very tight tolerance."""
    best_r = _calc_min_radius(points)
    best_points = points.copy()
    
    # Use the original gradient method with very small steps
    step_size = 1e-6
    for it in range(10000):
        if time.time() > deadline:
            break
        
        grad, current_r = _compute_gradient(points, current_r)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        grad /= grad_norm
        
        new_points = points + step_size * grad
        new_r = _calc_min_radius(new_points)
        
        if new_r > current_r + 1e-16:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            step_size = min(step_size * 1.01, 1e-4)
        else:
            step_size *= 0.95
            if step_size < 1e-12:
                break
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = 110.0 # 120s limit, keep margin
    
    # 1. Initialization: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
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
        spacing *= 0.98
        
    current_r = _calc_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Phase 1: Original gradient-based local search
    step_size = 1e-4
    phase1_deadline = start_time + 30.0  # Spend ~30s on phase 1
    
    while time.time() - start_time < phase1_deadline:
        grad, current_r = _compute_gradient(points, current_r)
        
        if np.linalg.norm(grad) < 1e-10:
            break
            
        # Try a step
        new_points = points + step_size * grad
        
        new_r = _calc_min_radius(new_points)
        
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
            step_size *= 1.05 # Increase step if improving
            if step_size > 1e-2:
                step_size = 1e-2
        else:
            step_size *= 0.8 # Decrease step if not improving
            if step_size < 1e-7:
                break
                
    # Phase 2: Smooth soft-min optimization
    phase2_deadline = start_time + 85.0  # Spend ~55s on phase 2
    if time.time() < phase2_deadline:
        points, current_r = _smooth_optimize(points, current_r, phase2_deadline, max_iters=5000)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
    
    # Phase 3: High-precision polish
    phase3_deadline = deadline - 2.0  # Leave 2s margin
    if time.time() < phase3_deadline:
        points, current_r = _polish(best_points, phase3_deadline)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
    
    return best_points
