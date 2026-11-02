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
    
    # Distribute points into rings based on circumference
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

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9
    
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

def _epigraph_objective_and_grad(x, beta):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    Objective: -t + penalty for violations, with smooth approximation.
    x = [points (2N), t (1)]
    """
    p = x[:2*N].reshape(-1, 2)
    t = x[2*N]
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values: d_ij/2 for pairs, 1-|p_i| for boundary
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min of (all_vals - t)
    violations = all_vals - t
    r_soft, weights = _soft_min_and_grad(violations, beta)
    
    # Objective: -t - penalty * max(0, -r_soft)
    # If r_soft < 0, some constraints violated
    penalty = max(0.0, -r_soft)
    objective = -t - 100.0 * penalty
    
    # Gradient with respect to points
    grad_p = np.zeros_like(p)
    
    # Pair contributions (derivative of soft-min w.r.t. pair distances)
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad_p, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad_p, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad_p += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    # Gradient with respect to t
    grad_t = -1.0
    if penalty > 0:
        # Additional penalty gradient
        grad_t += 100.0  # d/dt of -100*max(0,-r_soft) when violated
    
    grad = np.concatenate([grad_p.flatten(), [grad_t]])
    
    return objective, grad

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
    
    # Constraint values: d_ij/2 for pairs, 1-|p_i| for boundary
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

def _local_search_polish(points, time_limit=10.0, step_size=0.001):
    """
    Direct local search with adaptive step size: try moving each point in small steps,
    accept if it improves the minimum constraint. Uses gradient-informed directions.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    n = len(current_points)
    
    # Base directions
    directions = [
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
    ]
    
    # Compute gradient-based directions for critical points
    diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    norms = np.linalg.norm(current_points, axis=1)
    
    min_pair = np.min(dists) / 2.0
    min_boundary = np.min(1.0 - norms)
    
    # Gradient-based movement: move points away from their closest neighbors
    grad_dirs = np.zeros_like(current_points)
    for i in range(n):
        # Find nearest neighbor
        min_d_idx = np.argmin(dists[i])
        if dists[i, min_d_idx] > 1e-10:
            dir_vec = current_points[i] - current_points[min_d_idx]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-10:
                grad_dirs[i] += dir_vec / dir_norm
        # Away from boundary if close
        if norms[i] > 0.9:
            grad_dirs[i] -= current_points[i] / max(norms[i], 1e-10)
    
    # Normalize gradient directions
    for i in range(n):
        gn = np.linalg.norm(grad_dirs[i])
        if gn > 1e-10:
            grad_dirs[i] /= gn
    
    # Add gradient-informed directions
    extra_dirs = []
    for i in range(n):
        extra_dirs.append(grad_dirs[i])
    
    improved = True
    step = step_size
    while improved and time.time() - start_time < time_limit:
        improved = False
        step *= 0.95  # Gradually reduce step size
        for _ in range(30):  # Max passes per iteration
            if time.time() - start_time > time_limit:
                break
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                # Try base directions
                for d in directions:
                    trial = current_points.copy()
                    trial[i] += step * np.array(d)
                    norm = np.linalg.norm(trial[i])
                    if norm > 1.0:
                        trial[i] = trial[i] / norm * (1.0 - 1e-10)
                    
                    trial_score = _compute_score(trial)
                    if trial_score > current_score + 1e-14:
                        current_points = trial
                        current_score = trial_score
                        improved = True
                        break
                # Try gradient-informed direction
                if not improved:
                    trial = current_points.copy()
                    trial[i] += step * grad_dirs[i]
                    norm = np.linalg.norm(trial[i])
                    if norm > 1.0:
                        trial[i] = trial[i] / norm * (1.0 - 1e-10)
                    
                    trial_score = _compute_score(trial)
                    if trial_score > current_score + 1e-14:
                        current_points = trial
                        current_score = trial_score
                        improved = True
                        break
    
    return current_points, current_score

def _local_optimize(points, time_limit=30.0, beta_schedule=None, max_iter=500, use_epigraph=False):
    """
    Local optimization using L-BFGS-B with soft-min or epigraph objective and adaptive beta.
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
            if use_epigraph:
                # Epigraph formulation: x = [points, t]
                init_t = current_score
                x0 = np.concatenate([current_points.flatten(), [init_t]])
                res = minimize(
                    lambda x: _epigraph_objective_and_grad(x, beta)[0],
                    x0,
                    jac=lambda x: _epigraph_objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': max_iter,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                        'maxls': 50,
                        'disp': False
                    }
                )
                candidate = res.x[:2*N].reshape(-1, 2)
            else:
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Use hex lattice as primary initialization (proven good starting point)
    best_points = _hex_lattice_init(N)
    best_score = _compute_score(best_points)
    
    # Try ring initialization as alternative
    ring_points = _ring_decomposition_init(N, 10, 0.90)
    ring_score = _compute_score(ring_points)
    if ring_score > best_score:
        best_points = ring_points
        best_score = ring_score
    
    # Phase 1: L-BFGS optimization with moderate beta (soft-min)
    if time.time() < deadline - 70:
        remaining_time = min(45.0, deadline - 70 - time.time())
        best_points, best_score = _local_optimize(
            best_points, 
            time_limit=remaining_time,
            beta_schedule=[50.0, 200.0, 1000.0],
            max_iter=150
        )
    
    # Phase 2: Epigraph formulation for better constraint handling
    if time.time() < deadline - 40:
        remaining_time = min(25.0, deadline - 40 - time.time())
        best_points, best_score = _local_optimize(
            best_points,
            time_limit=remaining_time,
            beta_schedule=[500.0, 2000.0],
            max_iter=100,
            use_epigraph=True
        )
    
    # Phase 3: Higher beta for finer optimization (soft-min)
    if time.time() < deadline - 15:
        remaining_time = min(15.0, deadline - 15 - time.time())
        best_points, best_score = _local_optimize(
            best_points,
            time_limit=remaining_time,
            beta_schedule=[5000.0, 10000.0],
            max_iter=100
        )
    
    # Phase 4: Direct local search polish with adaptive steps
    if time.time() < deadline - 5:
        remaining_time = min(10.0, deadline - 5 - time.time())
        best_points, best_score = _local_search_polish(
            best_points,
            time_limit=remaining_time,
            step_size=0.001
        )
    
    # Phase 5: Final high-precision polish with very small steps
    if time.time() < deadline - 1:
        remaining_time = min(4.0, deadline - 1 - time.time())
        best_points, best_score = _local_search_polish(
            best_points,
            time_limit=remaining_time,
            step_size=0.0001
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
