"""Improved program for packing 300 circles in a unit circle using rigid-body mechanics with spring constraints."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    Tries multiple spacings to find the best initial packing.
    """
    best_pts = None
    best_score = -1.0
    
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        
        nx = int(2.0 / spacing) + 4
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 4
        
        x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
        
        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0 + 1e-9:
                    pts_list.append([px, py])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        score = _compute_score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random initialization
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    
    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    
    return best_pts

def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """
    Initialize points in concentric rings with optimized radial distances.
    """
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3
    
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Ring radii
    ring_radii = [ring_spread * (i + 1) / (num_rings + 1) for i in range(num_rings)]
    
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

def _spring_mechanics_objective(x, target_r, k_spring=100.0, beta=1000.0):
    """
    Rigid-body mechanics objective: minimize energy from spring violations + soft-min clearance.
    x: flattened positions
    target_r: target circle radius (half the minimum separation)
    k_spring: spring stiffness for maintaining separation
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    dists = cdist(p, p)
    np.fill_diagonal(dists, np.inf)
    
    # Spring energy: penalize pairs closer than 2*target_r
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Spring force: Hooke's law for compression
    spring_violations = np.maximum(2.0 * target_r - pair_dists, 0.0)
    spring_energy = 0.5 * k_spring * np.sum(spring_violations**2)
    
    # Boundary clearance
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Soft-min of boundary clearance
    r_soft, _ = _soft_min_and_grad(boundary_dists, beta)
    
    # Combined objective: spring energy - clearance
    objective = spring_energy - r_soft
    
    return objective

def _spring_mechanics_grad(x, target_r, k_spring=100.0, beta=1000.0):
    """
    Gradient of the spring mechanics objective.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    dists = cdist(p, p)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    i_idx, j_idx = iu
    
    # Spring gradient
    grad = np.zeros_like(p)
    
    # Only consider compressed springs
    compressed = pair_dists < 2.0 * target_r
    if np.any(compressed):
        comp_i = i_idx[compressed]
        comp_j = j_idx[compressed]
        comp_d = pair_dists[compressed]
        
        vecs = p[comp_i] - p[comp_j]
        safe_d = np.where(comp_d > 1e-12, comp_d, 1e-12)
        
        # Force magnitude: k * (2r - d) * (d / d) = k * (2r - d)
        mag = k_spring * (2.0 * target_r - comp_d)
        
        np.add.at(grad, comp_i, mag[:, np.newaxis] * vecs / safe_d[:, np.newaxis])
        np.add.at(grad, comp_j, -mag[:, np.newaxis] * vecs / safe_d[:, np.newaxis])
    
    # Boundary gradient
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Soft-min gradient for boundary
    _, weights = _soft_min_and_grad(boundary_dists, beta)
    
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += weights[:, np.newaxis] * bound_grad_vecs
    
    # Total gradient (negative because we're maximizing clearance)
    grad_obj = grad
    
    return grad_obj

def _local_optimize_spring(points, time_limit=30.0, target_r=None, k_spring=50.0, 
                           beta_schedule=None, max_iter=200):
    """
    Local optimization using L-BFGS-B with spring mechanics objective.
    """
    start_time = time.time()
    
    if target_r is None:
        target_r = _compute_score(points)
    
    if beta_schedule is None:
        beta_schedule = [100.0, 500.0, 2000.0, 10000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            opts = {
                'maxiter': max_iter,
                'gtol': 1e-10,
                'ftol': 1e-14,
                'maxls': 100,
                'disp': False
            }
            
            res = minimize(
                lambda x: _spring_mechanics_objective(x, target_r, k_spring, beta),
                current_points.flatten(),
                jac=lambda x: _spring_mechanics_grad(x, target_r, k_spring, beta),
                method='L-BFGS-B',
                options=opts
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                # Update target_r to current best
                target_r = current_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _local_optimize_softmin(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """
    Local optimization using L-BFGS-B with soft-min objective (original method).
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
            if beta >= 10000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 100,
                    'disp': False
                }
            elif beta >= 1000:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-9,
                    'ftol': 1e-13,
                    'maxls': 80,
                    'disp': False
                }
            else:
                opts = {
                    'maxiter': max_iter,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options=opts
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                
        except Exception:
            continue
    
    return current_points, current_score

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    dists = cdist(p, p)
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

def _force_directed_refine(points, iterations=300, time_limit=10.0):
    """
    Force-directed local optimization: push apart overlapping circles,
    pull toward boundary to maximize radius.
    """
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current minimum radius
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - norms)
        r = min(min_pair, min_boundary)
        
        # Build forces
        forces = np.zeros_like(points)
        
        # Pair repulsion for close pairs
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        i_idx, j_idx = iu
        
        # Only consider pairs within 2*r * 1.5
        close_mask = pair_dists < 2.0 * r * 1.5
        if np.any(close_mask):
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_d = pair_dists[close_mask]
            vecs = points[close_i] - points[close_j]
            safe_d = np.where(close_d > 1e-10, close_d, 1e-10)
            # Force magnitude: stronger for closer pairs
            mag = (2.0 * r - close_d) / safe_d
            forces[close_i] += mag[:, np.newaxis] * vecs
            forces[close_j] -= mag[:, np.newaxis] * vecs
        
        # Boundary attraction
        if min_boundary <= min_pair:
            for i in range(n):
                norm_i = norms[i]
                if norm_i > 0.01:
                    radial_dir = points[i] / norm_i
                    if 1.0 - norm_i < r * 1.1:
                        forces[i] -= radial_dir * 0.01
                    else:
                        forces[i] += radial_dir * 0.005
        
        # Apply forces with adaptive step size
        step = 0.1 * r / (1.0 + it * 0.01)
        new_points = points + step * forces
        
        # Clamp to unit circle
        new_norms = np.linalg.norm(new_points, axis=1)
        for i in range(n):
            if new_norms[i] > 1.0 - 1e-10:
                new_points[i] = new_points[i] / new_norms[i] * (1.0 - 1e-10)
        
        # Check for improvement
        new_score = _compute_score(new_points)
        if new_score >= r - 1e-8:
            points = new_points
    
    return points, _compute_score(points)

def _global_rotation_search(points, n_angles=36, time_limit=5.0):
    """
    Try rotating the entire configuration by various angles to find better alignment.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for angle in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
        if time.time() - start_time > time_limit:
            break
        
        theta = angle
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        
        rotated = points @ rot.T
        score = _compute_score(rotated)
        
        if score > best_score:
            best_score = score
            best_points = rotated
    
    return best_points, best_score

def _radial_displacement_refine(points, time_limit=10.0):
    """
    Refine by adjusting radial positions of boundary circles.
    """
    start_time = time.time()
    n = len(points)
    
    # Identify boundary circles (those with norm > 0.7)
    norms = np.linalg.norm(points, axis=1)
    boundary_mask = norms > 0.7
    
    if not np.any(boundary_mask):
        return points, _compute_score(points)
    
    boundary_indices = np.where(boundary_mask)[0]
    n_boundary = len(boundary_indices)
    
    # Optimize radial displacements for boundary circles
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Try small radial adjustments
    for delta_r in np.linspace(-0.02, 0.02, 21):
        if time.time() - start_time > time_limit:
            break
        
        candidate = points.copy()
        for idx in boundary_indices:
            norm_i = norms[idx]
            if norm_i > 1e-6:
                radial_dir = candidate[idx] / norm_i
                candidate[idx] += delta_r * radial_dir
        
        # Clamp to unit circle
        new_norms = np.linalg.norm(candidate, axis=1)
        for i in range(n):
            if new_norms[i] > 1.0 - 1e-10:
                candidate[i] = candidate[i] / new_norms[i] * (1.0 - 1e-10)
        
        score = _compute_score(candidate)
        if score > best_score:
            best_score = score
            best_points = candidate
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try multiple initializations
    initializations = [
        ("hex", _hex_lattice_init(N)),
        ("ring_8", _ring_decomposition_init(N, 8, 0.92)),
        ("ring_10", _ring_decomposition_init(N, 10, 0.90)),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for name, points in initializations:
        if time.time() > deadline - 60:
            break
        
        score = _compute_score(points)
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Stage 1: Global rotation search
    if time.time() < deadline - 100:
        remaining_time = min(5.0, deadline - 100 - time.time())
        if remaining_time > 1.0:
            best_points, best_score = _global_rotation_search(best_points, n_angles=36, time_limit=remaining_time)
    
    # Stage 2: Spring mechanics optimization (coarse)
    if time.time() < deadline - 80:
        remaining_time = min(25.0, deadline - 80 - time.time())
        if remaining_time > 5.0:
            best_points, best_score = _local_optimize_spring(
                best_points,
                time_limit=remaining_time,
                k_spring=50.0,
                beta_schedule=[100.0, 500.0, 2000.0],
                max_iter=150
            )
    
    # Stage 3: Soft-min optimization (coarse to fine)
    if time.time() < deadline - 50:
        remaining_time = min(30.0, deadline - 50 - time.time())
        if remaining_time > 10.0:
            best_points, best_score = _local_optimize_softmin(
                best_points,
                time_limit=remaining_time,
                beta_schedule=[10.0, 50.0, 200.0, 1000.0],
                max_iter=200
            )
    
    # Stage 4: Radial displacement refinement
    if time.time() < deadline - 30:
        remaining_time = min(10.0, deadline - 30 - time.time())
        if remaining_time > 3.0:
            best_points, best_score = _radial_displacement_refine(best_points, time_limit=remaining_time)
    
    # Stage 5: Spring mechanics optimization (fine)
    if time.time() < deadline - 20:
        remaining_time = min(15.0, deadline - 20 - time.time())
        if remaining_time > 5.0:
            best_points, best_score = _local_optimize_spring(
                best_points,
                time_limit=remaining_time,
                k_spring=200.0,
                beta_schedule=[5000.0, 20000.0],
                max_iter=200
            )
    
    # Stage 6: Force-directed refinement
    if time.time() < deadline - 10:
        remaining_time = min(8.0, deadline - 10 - time.time())
        if remaining_time > 2.0:
            fd_points, fd_score = _force_directed_refine(
                best_points,
                iterations=300,
                time_limit=remaining_time
            )
            if fd_score > best_score:
                best_points = fd_points
                best_score = fd_score
    
    # Stage 7: Final high-precision polish
    if time.time() < deadline - 3:
        remaining_time = min(5.0, deadline - 3 - time.time())
        if remaining_time > 1.0:
            best_points, best_score = _local_optimize_softmin(
                best_points,
                time_limit=remaining_time,
                beta_schedule=[50000.0, 200000.0],
                max_iter=300
            )
    
    # Final validation and clamping
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-12)
        best_score = _compute_score(best_points)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.9, neginf=-0.9)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
