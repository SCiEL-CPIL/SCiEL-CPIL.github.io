"""Improved program for packing 300 circles in a unit circle using radial-ring decomposition with smooth soft-min optimization."""
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

def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """
    Initialize points in concentric rings with optimized radial distances.
    Each ring gets a proportional number of points based on circumference.
    """
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3
    
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Estimate circle radius from area
    r_est = 1.0 / np.sqrt(n)
    
    # Ring radii - use non-uniform spacing for better packing
    ring_radii = []
    for i in range(num_rings):
        frac = (i + 1) / (num_rings + 1)
        rr = ring_spread * np.sqrt(frac)
        ring_radii.append(rr)
    
    # Distribute points proportionally to circumference
    circumferences = [2.0 * np.pi * rr for rr in ring_radii]
    total_circ = sum(circumferences)
    
    ring_counts = []
    for i, circ in enumerate(circumferences):
        count = int(remaining * circ / total_circ)
        ring_counts.append(max(1, count))
    
    # Adjust to get exact count
    diff = remaining - sum(ring_counts)
    if diff > 0:
        for _ in range(diff):
            ring_counts[-1] += 1
    elif diff < 0:
        for i in range(len(ring_counts) - 1, -1, -1):
            if ring_counts[i] > 1 and -diff > 0:
                ring_counts[i] -= 1
                diff += 1
    
    # Place points in each ring with angular offsets
    for i, (count, rr) in enumerate(zip(ring_counts, ring_radii)):
        if count == 0:
            continue
        d_theta = 2.0 * np.pi / count
        offset = np.pi / count * (i % 2)
        for k in range(count):
            theta = k * d_theta + offset
            points.append([rr * np.cos(theta), rr * np.sin(theta)])
    
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    best_pts = None
    best_score = -1.0
    
    for scale in [0.88, 0.92, 0.96, 1.0, 1.04, 1.08]:
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
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(boundary_dists)
        score = min(min_pair_dist, min_boundary)
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    
    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    
    return best_pts

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    """
    values = np.array(values)
    z = -beta * values
    
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    grad_obj = -grad
    
    return objective, grad_obj

def _local_optimize(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """
    Local optimization using L-BFGS-B with soft-min objective and adaptive beta.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _local_optimize_fast(points, time_limit=30.0, beta_schedule=None, max_iter=300):
    """
    Faster local optimization with fewer beta stages for tighter time budget.
    """
    start_time = time.time()
    
    if beta_schedule is None:
        beta_schedule = [20.0, 100.0, 500.0, 2000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-11,
                    'maxls': 40,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _force_directed_refine(points, time_limit=10.0, n_iter=200):
    """
    Force-directed refinement: push apart close pairs and pull inward near boundary.
    Uses vectorized operations for speed.
    """
    start_time = time.time()
    p = points.copy()
    n = len(p)
    
    step = 0.005
    best_score = _compute_score(p)
    r_est = 1.0 / np.sqrt(n)
    threshold = 3.5 * r_est
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces vectorized
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1e-10)
        
        # Mask for relevant pairs
        mask = (dists < threshold) & (dists > 1e-10)
        
        # Compute repulsive forces
        forces = np.zeros_like(p)
        # f_ij = 0.1 / d^2 in direction (p_i - p_j)/d
        inv_d2 = np.where(mask, 0.1 / (dists**2), 0.0)
        inv_d = np.where(mask, 1.0 / dists, 0.0)
        
        # Sum forces on each point
        # force on i from j: inv_d2[i,j] * inv_d[i,j] * (p[i] - p[j])
        # We can compute: for each i, sum over j of inv_d2[i,j] * (p[i] - p[j]) / dists[i,j]
        # = p[i] * sum_j inv_d2[i,j]/dists[i,j] - sum_j inv_d2[i,j]*p[j]/dists[i,j]
        
        weight = inv_d2 * inv_d  # 0.1 / d^3
        weight_sum = np.sum(weight, axis=1)
        weighted_p = weight @ p  # shape (n, 2)
        
        forces = weight_sum[:, np.newaxis] * p - weighted_p
        
        # Boundary attraction
        norms = np.linalg.norm(p, axis=1)
        boundary_mask = norms > 0.93
        if np.any(boundary_mask):
            f_bound = 0.03 * (norms[boundary_mask] - 0.93)
            forces[boundary_mask] -= f_bound[:, np.newaxis] * p[boundary_mask] / norms[boundary_mask, np.newaxis]
        
        # Apply forces
        p_new = p + step * forces
        score_new = _compute_score(p_new)
        
        if score_new > best_score:
            p = p_new
            best_score = score_new
            step = min(step * 1.05, 0.02)
        else:
            step *= 0.92
    
    return p, best_score

def _final_polish(points, time_limit=15.0):
    """
    Final high-precision polish pass with very high beta and tight tolerances.
    """
    start_time = time.time()
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # First do force-directed refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        current_points, current_score = _force_directed_refine(
            current_points, 
            time_limit=min(remaining * 0.4, 8.0)
        )
    
    # Then use very high beta for precise local optimization
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        for beta in [3000.0, 8000.0]:
            remaining = time_limit - (time.time() - start_time)
            if remaining < 2:
                break
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, beta)[0],
                    current_points.flatten(),
                    jac=lambda x: _objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 400,
                        'gtol': 1e-10,
                        'ftol': 1e-13,
                        'maxls': 100,
                        'disp': False
                    }
                )
                
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > current_score:
                    current_points = candidate
                    current_score = candidate_score
                    
            except Exception:
                pass
    
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations
    initializations = [
        ("hex", _hex_lattice_init(N)),
        ("ring_8", _ring_decomposition_init(N, 8, 0.92)),
        ("ring_10", _ring_decomposition_init(N, 10, 0.90)),
        ("ring_12", _ring_decomposition_init(N, 12, 0.95)),
        ("ring_14", _ring_decomposition_init(N, 14, 0.93)),
    ]
    
    # Evaluate and sort initializations by score
    scored_inits = []
    for name, points in initializations:
        score = _compute_score(points)
        scored_inits.append((score, points))
        
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    best_points = None
    best_score = -np.inf
    
    # Optimize the top 2 initializations with fast optimizer
    num_optimizations = min(2, len(scored_inits))
    
    for i in range(num_optimizations):
        if time.time() > deadline - 45:
            break
            
        init_score, points = scored_inits[i]
        
        remaining_time = deadline - time.time() - 25
        time_per_opt = remaining_time / (num_optimizations - i)
        time_per_opt = min(time_per_opt, 50.0)
        
        if time_per_opt < 12:
            break
            
        optimized_points, opt_score = _local_optimize_fast(
            points, 
            time_limit=time_per_opt,
            beta_schedule=[15.0, 80.0, 400.0, 2500.0],
            max_iter=500
        )
        
        if opt_score > best_score:
            best_points = optimized_points
            best_score = opt_score
    
    if best_points is None:
        best_points = scored_inits[0][1]
        best_score = scored_inits[0][0]
    
    # Final polish on the best result
    if time.time() < deadline - 18:
        remaining_time = deadline - time.time() - 12
        if remaining_time > 10:
            best_points, best_score = _final_polish(
                best_points,
                time_limit=remaining_time
            )
            
    # Final validation
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-10)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
