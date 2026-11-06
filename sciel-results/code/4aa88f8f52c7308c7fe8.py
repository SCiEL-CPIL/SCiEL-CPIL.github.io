"""Packing 300 equal circles in a unit circle.

Strategy:
1. Generate a Fibonacci-sphere-inspired initial configuration (uniform, non-lattice).
2. Refine with a vectorised repulsion-based relaxation using a soft penalty for
   overlaps and boundary violations.
3. Polish with a local search that uses exact radius computation.
4. Deadline guard ensures we return the best configuration before the 150 s kill.
"""
import numpy as np
import time

N = 300


def _compute_radius(pts):
    """Compute the radius for a configuration of N points."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances using vectorized approach
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, boundary_min)


def _compute_radius_fast(pts):
    """Faster radius computation using KDTree nearest neighbours."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0:
        return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_pair / 2.0, boundary_min)


def _fibonacci_disk(n_points=N):
    """
    Generate a uniform distribution of points on the disk using a
    Fibonacci-sphere-inspired mapping.
    """
    # Use the Fibonacci sphere construction to get well-distributed points on
    # a unit sphere, then project onto the disk via stereographic-like mapping.
    # Actually, a better approach for the disk: use the golden-angle spiral.
    # The golden angle is 2*pi*(1 - 1/phi) where phi = (1+sqrt(5))/2
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)
    
    # Place points using a sunflower/Fibonacci pattern
    # r_k = sqrt((k + 0.5) / N) to avoid clustering at centre
    # theta_k = k * golden_angle
    k = np.arange(n_points, dtype=float)
    r = np.sqrt((k + 0.5) / n_points)
    theta = k * golden_angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Slightly shrink to keep inside the unit circle with margin
    scale = 0.995
    x *= scale
    y *= scale
    
    return np.column_stack([x, y])


def _hex_initial(spacing, n_points=N):
    """Generate hexagonal lattice points and select the best n_points."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 3
    
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            r = np.hypot(x, y)
            if r < 1.0:
                points_list.append((x, y))
    
    if len(points_list) < n_points:
        return None, 0.0
    
    points = np.array(points_list)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n_points]
    pts = points[idx]
    r = _compute_radius(pts)
    return pts, r


def _repulsion_relax(pts, deadline, seed=123):
    """
    Repulsion-based relaxation using a soft penalty for overlaps and
    boundary violations.
    """
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    # Parameters
    target_spacing = 2.0 * max(0.0532, current_r) * 1.005
    force_scale = 0.004
    boundary_scale = 0.001
    damping = 0.8
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Build KDTree for efficient neighbour queries
        tree = cKDTree(current)
        
        # Find close pairs (distance < target_spacing)
        pairs = tree.query_pairs(target_spacing, output_type='ndarray')
        
        if len(pairs) == 0:
            # No close pairs; check boundary
            dists_origin = np.hypot(current[:, 0], current[:, 1])
            if np.max(dists_origin) < 0.98:
                break
            else:
                # Only boundary forces
                forces = np.zeros_like(current)
                near_boundary = dists_origin > 0.95
                for k_idx in np.where(near_boundary)[0]:
                    r_mag = dists_origin[k_idx]
                    if r_mag > 1e-10:
                        bf = boundary_scale * (r_mag - 0.95) / 0.05
                        forces[k_idx] -= (current[k_idx] / r_mag) * bf
                current += forces * damping
                # Keep inside circle
                r_mags = np.hypot(current[:, 0], current[:, 1])
                mask = r_mags > 0.998
                if np.any(mask):
                    current[mask] = current[mask] / r_mags[mask, np.newaxis] * 0.998
                new_r = _compute_radius_fast(current)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best = current.copy()
                continue
        
        # Accumulate forces from close pairs
        forces = np.zeros_like(current)
        
        for i, j in pairs:
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = rng.randn(2)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            d = norm
            # 1/d^2 weighted force, capped
            force_mag = force_scale * (target_spacing / max(d, 0.01)) ** 2
            force_mag = min(force_mag, 0.02)
            
            forces[i] += direction * force_mag
            forces[j] -= direction * force_mag
        
        # Add boundary repulsion
        dists_origin = np.hypot(current[:, 0], current[:, 1])
        near_boundary = dists_origin > 0.95
        for k_idx in np.where(near_boundary)[0]:
            r_mag = dists_origin[k_idx]
            if r_mag > 1e-10:
                bf = boundary_scale * (r_mag - 0.95) / 0.05
                forces[k_idx] -= (current[k_idx] / r_mag) * bf
        
        # Apply forces with damping
        current += forces * damping
        
        # Keep inside circle
        r_mags = np.hypot(current[:, 0], current[:, 1])
        mask = r_mags > 0.998
        if np.any(mask):
            current[mask] = current[mask] / r_mags[mask, np.newaxis] * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best = current.copy()
        
        # Adaptive parameters
        if iteration % 200 == 0 and iteration > 0:
            # Slightly reduce force to avoid oscillation
            force_scale *= 0.95
            damping *= 0.98
    
    return best, best_r


def _local_search(pts, deadline, seed=42, use_fast=True):
    """Local search: perturb random points, accept if radius improves."""
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    current = pts.copy()
    compute_r = _compute_radius_fast if use_fast else _compute_radius
    current_r = compute_r(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.005
    patience = 0
    max_patience = 2000
    
    while time.time() < deadline and patience < max_patience:
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        nn_dists = dists[:, 1]
        
        min_nn_idx = np.argmin(nn_dists)
        min_nn_dist = nn_dists[min_nn_idx]
        
        dists_origin = np.hypot(current[:, 0], current[:, 1])
        max_boundary_idx = np.argmax(dists_origin)
        boundary_proximity = dists_origin[max_boundary_idx]
        
        roll = rng.random()
        if min_nn_dist < 2 * current_r * 1.03 and roll < 0.7:
            idx = min_nn_idx
        elif boundary_proximity > 0.97 and roll < 0.9:
            idx = max_boundary_idx
        else:
            idx = rng.randint(N)
        
        old_pos = current[idx].copy()
        
        best_new_r = current_r
        best_new_pos = old_pos.copy()
        
        n_attempts = 8 if patience < 300 else 5
        for attempt in range(n_attempts):
            base_step = step_size
            nn_idx = tree.query(current[idx], k=2)[1][0]
            if nn_idx != idx:
                direction_away = current[idx] - current[nn_idx]
                norm = np.linalg.norm(direction_away)
                if norm > 1e-10:
                    direction_away = direction_away / norm
                    new_pos = old_pos + base_step * (0.4 * rng.uniform(-1, 1, 2) + 0.6 * direction_away)
                else:
                    new_pos = old_pos + base_step * rng.uniform(-1, 1, 2)
            else:
                new_pos = old_pos + base_step * rng.uniform(-1, 1, 2)
            
            new_r_mag = np.hypot(new_pos[0], new_pos[1])
            if new_r_mag > 0.997:
                new_pos = new_pos / new_r_mag * 0.997
            
            current[idx] = new_pos
            new_r = compute_r(current)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos.copy()
        
        current[idx] = best_new_pos
        
        if best_new_r > current_r + 1e-13:
            current_r = best_new_r
            patience = 0
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            current[idx] = old_pos
            patience += 1
            if patience % 25 == 0:
                step_size *= 0.93
            if patience % 120 == 0 and step_size < 0.0015:
                step_size = 0.012
            if step_size < 1e-7:
                break
    
    return best, best_r


def _soft_penalty_relax(pts, deadline, seed=777):
    """
    Repulsion-based relaxation using a smooth soft penalty (log-sum-exp)
    for overlaps and boundary violations.
    """
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    current = pts.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    beta = 50.0  # Sharpness of soft-min
    gamma = 80.0  # Sharpness for boundary term
    lr = 0.003
    n_iter = 2000
    
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute soft penalty
        # Pairwise distances
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of pairwise distances
        iu = np.triu_indices(N, k=1)
        dists_sq_iu = dists_sq[iu]
        
        # To avoid numerical issues, clip very small distances
        dists_sq_iu = np.maximum(dists_sq_iu, 1e-12)
        
        exp_vals = np.exp(-beta * dists_sq_iu)
        sum_exp = np.sum(exp_vals)
        
        # Gradient of soft-min squared distance
        # d(soft_min_sq)/dx = (1/sum_exp) * sum_j (-2*beta) * (x_i - x_j) * exp(-beta * d_ij^2)
        # For each pair (i,j), contribution to grad[i] is (-2*beta)*(x_i-x_j)*w_ij
        # and to grad[j] is (+2*beta)*(x_i-x_j)*w_ij
        
        weights = exp_vals / sum_exp
        weights_matrix = np.zeros((N, N))
        weights_matrix[iu] = weights
        weights_matrix[(iu[1], iu[0])] = weights  # symmetric
        
        # Gradient of soft_min_sq
        grad_soft_min_sq = np.zeros_like(current)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = weights_matrix[i, j]
                if w > 1e-15:
                    grad_soft_min_sq[i] += -2.0 * beta * w * (current[i] - current[j])
        
        # Convert to gradient of soft_min_dist
        soft_min_sq = -np.log(sum_exp) / beta
        soft_min_dist = np.sqrt(max(soft_min_sq, 1e-12))
        
        # Gradient of soft_min_dist w.r.t. x
        # d(soft_min_dist)/dx = (1/(2*sqrt(soft_min_sq))) * d(soft_min_sq)/dx
        if soft_min_sq > 1e-12:
            grad_soft_min_dist = grad_soft_min_sq / (2.0 * np.sqrt(soft_min_sq))
        else:
            grad_soft_min_dist = np.zeros_like(current)
        
        # Boundary term: soft max of distances from origin
        dists_origin = np.hypot(current[:, 0], current[:, 1])
        # Soft max using log-sum-exp
        exp_boundary = np.exp(gamma * dists_origin)
        sum_exp_boundary = np.sum(exp_boundary)
        soft_max_dist = np.log(sum_exp_boundary) / gamma
        
        # Gradient of soft_max_dist
        weights_boundary = exp_boundary / sum_exp_boundary
        grad_boundary = np.zeros_like(current)
        for i in range(N):
            w = weights_boundary[i]
            if w > 1e-15 and dists_origin[i] > 1e-10:
                grad_boundary[i] = (w / dists_origin[i]) * current[i]
        
        # Combined objective: we want to maximize min(soft_min_dist/2, 1 - soft_max_dist)
        # Use soft-min of the two terms
        a = soft_min_dist / 2.0
        b = 1.0 - soft_max_dist
        gamma2 = 100.0
        exp_ab = np.exp(-gamma2 * np.array([a, b]))
        sum_ab = np.sum(exp_ab)
        soft_min_ab = -np.log(sum_ab) / gamma2
        
        # Gradient of soft_min_ab w.r.t. x
        # w_a = exp(-gamma2*a)/sum_ab, w_b = exp(-gamma2*b)/sum_ab
        w_a = np.exp(-gamma2 * a) / sum_ab
        w_b = np.exp(-gamma2 * b) / sum_ab
        
        # d(soft_min_ab)/dx = -w_a * (1/2) * d(soft_min_dist)/dx - w_b * (-1) * d(soft_max_dist)/dx
        # = -w_a/2 * grad_soft_min_dist + w_b * grad_boundary
        
        grad_total = -0.5 * w_a * grad_soft_min_dist + w_b * grad_boundary
        
        # Gradient ascent: move in direction of gradient
        # Scale step size
        step = lr * grad_total
        
        # Limit step size
        step_norms = np.hypot(step[:, 0], step[:, 1])
        max_step = 0.02
        for i in range(N):
            if step_norms[i] > max_step:
                step[i] = step[i] / step_norms[i] * max_step
        
        current += step
        
        # Keep inside circle
        r_mags = np.hypot(current[:, 0], current[:, 1])
        mask = r_mags > 0.998
        if np.any(mask):
            current[mask] = current[mask] / r_mags[mask, np.newaxis] * 0.998
        
        new_r = _compute_radius_fast(current)
        if new_r > best_r + 1e-13:
            best_r = new_r
            best = current.copy()
        
        # Adaptive learning rate
        if iteration % 100 == 0 and iteration > 0:
            lr *= 0.97
    
    return best, best_r


def construct():
    deadline = time.time() + 138.0  # safety margin before 150s kill
    
    # --- Stage 1: Generate multiple initial configurations ---
    best_points = None
    best_radius = 0.0
    
    # 1a. Fibonacci disk (uniform, non-lattice)
    fib_pts = _fibonacci_disk(N)
    fib_r = _compute_radius(fib_pts)
    if fib_r > best_radius:
        best_radius = fib_r
        best_points = fib_pts.copy()
    
    # 1b. Hexagonal lattice (multiple spacings)
    for trial in range(60):
        if time.time() > deadline - 100:
            break
        spacing = 0.076 + trial * 0.0008
        pts, r = _hex_initial(spacing)
        if pts is not None and r > best_radius:
            best_radius = r
            best_points = pts.copy()
    
    # 1c. Random initialization with some structure
    if time.time() < deadline - 95:
        rng_init = np.random.RandomState(0)
        for attempt in range(5):
            # Concentric rings
            n_rings = 10
            pts_list = []
            for ring in range(n_rings):
                r_ring = 0.05 + ring * 0.09
                n_in_ring = int(2 * np.pi * r_ring / 0.054) if ring > 0 else 1
                if n_in_ring < 1:
                    n_in_ring = 1
                angles = rng_init.uniform(0, 2 * np.pi, n_in_ring)
                for ang in angles:
                    x = r_ring * np.cos(ang)
                    y = r_ring * np.sin(ang)
                    if np.hypot(x, y) < 0.99:
                        pts_list.append((x, y))
            if len(pts_list) >= N:
                pts = np.array(pts_list[:N])
                r = _compute_radius(pts)
                if r > best_radius:
                    best_radius = r
                    best_points = pts.copy()
    
    if best_points is None:
        # Fallback
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_points *= 0.99
        best_radius = _compute_radius(best_points)
    
    # --- Stage 2: Repulsion relaxation ---
    if time.time() < deadline - 40:
        rep_budget = deadline - 35.0
        rep_pts, rep_r = _repulsion_relax(best_points, rep_budget, seed=123)
        if rep_r > best_radius:
            best_points = rep_pts
            best_radius = rep_r
    
    # --- Stage 3: Soft penalty relaxation ---
    if time.time() < deadline - 35:
        soft_budget = deadline - 30.0
        soft_pts, soft_r = _soft_penalty_relax(best_points, soft_budget, seed=777)
        if soft_r > best_radius:
            best_points = soft_pts
            best_radius = soft_r
    
    # --- Stage 4: Local search ---
    for seed in [42, 777, 999, 1234, 2024]:
        if time.time() > deadline - 20:
            break
        ls_budget = deadline - 15.0
        ls_pts, ls_r = _local_search(best_points, ls_budget, seed=seed, use_fast=True)
        if ls_r > best_radius:
            best_points = ls_pts
            best_radius = ls_r
    
    # --- Stage 5: Final polish with exact radius ---
    if time.time() < deadline - 10:
        polish_budget = deadline - 3.0
        polish_pts, polish_r = _local_search(best_points, polish_budget, seed=12345, use_fast=False)
        if polish_r > best_radius:
            best_points = polish_pts
            best_radius = polish_r
    
    # Final validation
    dists_origin = np.hypot(best_points[:, 0], best_points[:, 1])
    max_allowed = 1.0 - 1e-10
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_points[mask] *= (max_allowed / dists_origin[mask]) * 0.999
    
    # Ensure no coincident points
    from scipy.spatial import cKDTree
    tree = cKDTree(best_points)
    dists, pairs = tree.query(best_points, k=2)
    close_mask = dists[:, 1] < 1e-6
    if np.any(close_mask):
        close_indices = np.where(close_mask)[0]
        rng_fix = np.random.RandomState(999)
        for idx in close_indices:
            j = pairs[idx, 1]
            angle = rng_fix.uniform(0, 2 * np.pi)
            best_points[idx, 0] += 0.01 * np.cos(angle)
            best_points[idx, 1] += 0.01 * np.sin(angle)
            best_points[j, 0] -= 0.01 * np.cos(angle)
            best_points[j, 1] -= 0.01 * np.sin(angle)
    
    # Final check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        best_points *= 0.99
    
    return best_points
