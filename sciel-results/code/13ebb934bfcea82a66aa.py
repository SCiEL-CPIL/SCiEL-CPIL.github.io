"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
    Returns (soft_min_value, weights) where weights sum to 1.
    """
    values = np.asarray(values, dtype=np.float64)
    z = -beta * values
    
    max_z = np.max(z)
    # Clip z - max_z to avoid underflow issues for very negative values
    shifted = z - max_z
    shifted = np.clip(shifted, -700.0, 0.0)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta, iu):
    """
    Compute the negative soft-min objective and its gradient.
    Uses precomputed indices for efficiency.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances using precomputed indices
    i_idx, j_idx = iu
    vecs = p[i_idx] - p[j_idx]
    pair_dists_sq = np.sum(vecs**2, axis=1)
    pair_dists = np.sqrt(pair_dists_sq) / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_dists)]
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    # d(pair_dist)/dp_i = (p_i - p_j) / (2 * dist)  [since pair_dist = dist/2]
    # d(-beta*pair_dist)/dp_i = -beta * (p_i - p_j) / (2 * dist)
    # soft_min gradient: sum(w_k * d(-beta*c_k)/dp) / beta = sum(w_k * d(-c_k)/dp)
    # d(-pair_dist)/dp_i = -(p_i - p_j) / (2 * dist) = (p_j - p_i) / (2 * dist)
    # coeff for point i: w * (p_j - p_i) / (2 * dist) = w * (-vecs) / (2 * 2 * pair_dist)
    # Wait, pair_dist = dist/2, so dist = 2*pair_dist
    # (p_j - p_i) / (2 * dist) = -vecs / (2 * 2 * pair_dist) = -vecs / (4 * pair_dist)
    # Let's re-derive carefully.
    # c_k = |p_i - p_j| / 2
    # dc_k/dp_i = (p_i - p_j) / |p_i - p_j| * 1/2 = vecs / (2 * dist) = vecs / (4 * pair_dist)
    # d(-c_k)/dp_i = -vecs / (4 * pair_dist)
    # So grad contribution from pair k to p_i is w_k * (-vecs_k) / (4 * pair_dist_k)
    coeff = w_pairs / (4.0 * safe_pair_dists)
    
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_dists):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    # c_k = 1 - |p_k|
    # dc_k/dp_k = -p_k / |p_k|
    # d(-c_k)/dp_k = p_k / |p_k|
    bound_grad_vecs = p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return objective, grad

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Tries multiple spacings and rotations to find the best initial packing.
    """
    best_pts = None
    best_score = -1.0
    
    # Try different scales and rotations
    for scale in [0.95, 1.0, 1.05]:
        for rot_angle in [0.0, np.pi/12]:
            spacing = 2.0 / (np.sqrt(n) * scale)
            
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
            
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < n:
                continue
                
            pts = np.array(pts_list)
            
            # Apply rotation
            if rot_angle != 0.0:
                cos_r = np.cos(rot_angle)
                sin_r = np.sin(rot_angle)
                rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                pts = pts @ rot_mat.T
            
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:n]]
            
            # Compute score
            score = _compute_score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    
    # Add small deterministic perturbation to break symmetry
    np.random.seed(42)
    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    
    return best_pts

def _force_based_refine(points, max_iter=5000, tol=1e-10, deadline=None):
    """
    Force-based refinement: treat circles as charged particles with repulsion
    and attraction to maintain minimum separation. This handles hard constraints
    more directly than soft-min.
    """
    points = points.copy()
    n = len(points)
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pair repulsion: push apart if too close
        vecs = points[i_idx] - points[j_idx]
        dists = np.sqrt(np.sum(vecs**2, axis=1))
        
        # Target minimum distance (use current best estimate + margin)
        current_r = _compute_score(points)
        target_d = 2.0 * current_r * 1.01
        
        # Only apply force when below target
        below = dists < target_d
        if np.any(below):
            safe_d = np.where(dists[below] > 1e-12, dists[below], 1.0)
            # Force proportional to (target - dist) / dist
            force_mag = (target_d - dists[below]) / safe_d
            force_vecs = force_mag[:, np.newaxis] * vecs[below] / safe_d[:, np.newaxis]
            
            # Accumulate
            np.add.at(forces, i_idx[below], -force_vecs)
            np.add.at(forces, j_idx[below], force_vecs)
        
        # Boundary constraint: push inward if too close to boundary
        norms = np.linalg.norm(points, axis=1)
        boundary_dist = 1.0 - norms
        target_boundary = current_r * 1.01
        below_boundary = boundary_dist < target_boundary
        if np.any(below_boundary):
            safe_norms = np.where(norms[below_boundary] > 1e-12, norms[below_boundary], 1.0)
            force_mag = (target_boundary - boundary_dist[below_boundary]) / safe_norms
            force_vecs = force_mag[:, np.newaxis] * points[below_boundary] / safe_norms[:, np.newaxis]
            forces[below_boundary] -= force_vecs
        
        # Adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < tol:
            break
        step = min(0.1, tol / max_force)
        
        # Update
        points += step * forces
        
        # Clamp to boundary
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 30:
            break
        
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    # Annealed beta schedule for smoother convergence
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        try:
            # Use a tighter tolerance for higher betas
            if beta >= 20000:
                maxiter = 300
                gtol = 1e-12
                ftol = 1e-15
                maxls = 60
            elif beta >= 1000:
                maxiter = 400
                gtol = 1e-10
                ftol = 1e-13
                maxls = 50
            elif beta >= 100:
                maxiter = 300
                gtol = 1e-9
                ftol = 1e-12
                maxls = 40
            else:
                maxiter = 200
                gtol = 1e-8
                ftol = 1e-11
                maxls = 30
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta, iu)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta, iu)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'gtol': gtol,
                    'ftol': ftol,
                    'maxls': maxls,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 10:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass
    
    # Force-based refinement to handle hard constraints more directly
    if time.time() < deadline - 5:
        try:
            refined = _force_based_refine(best_points, max_iter=8000, tol=1e-12, deadline=deadline - 3)
            refined_score = _compute_score(refined)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
        except Exception:
            pass
    
    # Final high-precision polish with active set if time permits
    if time.time() < deadline - 3:
        try:
            # Identify active constraints (tightest pairs and boundary)
            current_r = best_score
            p = best_points
            i_idx, j_idx = iu
            vecs = p[i_idx] - p[j_idx]
            pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
            norms = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - norms
            
            # Active set: constraints within 1.5% of current minimum
            active_pairs = np.where(pair_dists < current_r * 1.015)[0]
            active_bounds = np.where(bound_dists < current_r * 1.015)[0]
            
            if len(active_pairs) + len(active_bounds) > 0:
                # Optimize only on active constraints with very tight tolerance
                def active_objective_and_grad(x):
                    pp = x.reshape(-1, 2)
                    
                    # Only compute for active constraints
                    if len(active_pairs) > 0:
                        a_vecs = pp[i_idx[active_pairs]] - pp[j_idx[active_pairs]]
                        a_pair_dists = np.sqrt(np.sum(a_vecs**2, axis=1)) / 2.0
                    else:
                        a_pair_dists = np.array([])
                        a_vecs = np.empty((0, 2))
                    
                    if len(active_bounds) > 0:
                        a_norms = np.linalg.norm(pp[active_bounds], axis=1)
                        a_bound_dists = 1.0 - a_norms
                    else:
                        a_bound_dists = np.array([])
                    
                    # Soft-min over active constraints only
                    beta_active = 100000.0
                    z_pairs = -beta_active * a_pair_dists if len(a_pair_dists) > 0 else np.array([-np.inf])
                    z_bounds = -beta_active * a_bound_dists if len(a_bound_dists) > 0 else np.array([-np.inf])
                    
                    all_z = np.concatenate([z_pairs, z_bounds])
                    max_z = np.max(all_z)
                    exp_all = np.exp(all_z - max_z)
                    sum_exp = np.sum(exp_all)
                    log_sum_exp = max_z + np.log(sum_exp)
                    val = log_sum_exp / beta_active
                    
                    grad = np.zeros_like(pp)
                    w_all = exp_all / sum_exp
                    
                    if len(active_pairs) > 0:
                        w_pairs = w_all[:len(a_pair_dists)]
                        safe_pd = np.where(a_pair_dists > 1e-12, a_pair_dists, 1.0)
                        coeff = w_pairs / (4.0 * safe_pd)
                        np.add.at(grad, i_idx[active_pairs], -coeff[:, np.newaxis] * a_vecs)
                        np.add.at(grad, j_idx[active_pairs], coeff[:, np.newaxis] * a_vecs)
                    
                    if len(active_bounds) > 0:
                        w_bounds = w_all[len(a_pair_dists):]
                        safe_bn = np.where(a_norms > 1e-12, a_norms, 1.0)
                        bound_grad_vecs = pp[active_bounds] / safe_bn[:, np.newaxis]
                        grad[active_bounds] += w_bounds[:, np.newaxis] * bound_grad_vecs
                    
                    return val, grad
                
                res = minimize(
                    lambda x: active_objective_and_grad(x)[0],
                    best_points.flatten(),
                    jac=lambda x: active_objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'gtol': 1e-14,
                        'ftol': 1e-16,
                        'maxls': 80,
                        'disp': False
                    }
                )
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
    
    # Final score
    best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
