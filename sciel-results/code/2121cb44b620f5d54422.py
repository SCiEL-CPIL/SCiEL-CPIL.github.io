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

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
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
        # If we didn't generate enough, shrink spacing slightly and retry
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
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    # Estimate ring structure
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    # Distribute points among rings
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to match total
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
    # Ensure we have exactly n points
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    # Place points
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
            phase = (k % 2) * np.pi / count
            for m in range(count):
                if idx >= n:
                    break
                angle = 2 * np.pi * m / count + phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    idx += 1
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _local_refine(points, iu, max_iter=50, tol=1e-10):
    """
    Local refinement using gradient descent with adaptive step size.
    Focus on the most constrained pairs to improve the minimum.
    """
    n = len(points)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)
    
    p = points.copy()
    current_score = _compute_score(p)
    
    for it in range(max_iter):
        # Compute all pair distances and boundary distances
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Find the minimum constraint
        min_pair_val = np.min(pair_dists)
        min_bound_val = np.min(bound_dists)
        
        # Compute gradient of the minimum (approximate by focusing on near-minimum constraints)
        # Use a threshold to identify "active" constraints
        threshold = 1.01 * min(min_pair_val, min_bound_val)
        
        grad = np.zeros_like(p)
        
        # Pair contributions for near-minimum pairs
        active_pairs = pair_dists < threshold
        if np.any(active_pairs):
            # Weight active pairs by inverse distance to emphasize the minimum
            weights = 1.0 / np.where(pair_dists[active_pairs] > 1e-12, pair_dists[active_pairs], 1.0)
            weights /= np.sum(weights)
            
            for k in range(len(weights)):
                idx = np.where(active_pairs)[0][k]
                i, j = i_idx[idx], j_idx[idx]
                w = weights[k]
                d = pair_dists[idx]
                if d > 1e-12:
                    # Move i away from j and j away from i
                    direction = vecs[idx] / (2.0 * d)
                    grad[i] -= w * direction
                    grad[j] += w * direction
        
        # Boundary contributions for near-minimum boundary points
        active_bounds = bound_dists < threshold
        if np.any(active_bounds):
            for i in np.where(active_bounds)[0]:
                if norms[i] > 1e-12:
                    # Move inward
                    grad[i] -= p[i] / norms[i]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        grad /= grad_norm
        
        # Adaptive step size: start with small step, increase if improving
        step = 1e-4
        improved = False
        for attempt in range(10):
            trial = p + step * grad
            trial_score = _compute_score(trial)
            if trial_score > current_score:
                p = trial
                current_score = trial_score
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try in the opposite direction
            for attempt in range(5):
                trial = p - step * grad
                trial_score = _compute_score(trial)
                if trial_score > current_score:
                    p = trial
                    current_score = trial_score
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
    
    return p, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 20:
            break
        
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        i_idx, j_idx = iu
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft min using log-sum-exp trick
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with increasing beta
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            if beta >= 5000:
                maxiter = 300
                gtol = 1e-12
                ftol = 1e-15
                maxls = 50
            elif beta >= 1000:
                maxiter = 400
                gtol = 1e-11
                ftol = 1e-14
                maxls = 40
            elif beta >= 100:
                maxiter = 350
                gtol = 1e-10
                ftol = 1e-13
                maxls = 35
            else:
                maxiter = 250
                gtol = 1e-9
                ftol = 1e-12
                maxls = 25
            
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
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

    # Final polish with very high beta if time permits
    if time.time() < deadline - 12:
        try:
            final_beta = 50000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'gtol': 1e-13,
                    'ftol': 1e-16,
                    'maxls': 60,
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

    # Local refinement pass to squeeze out more
    if time.time() < deadline - 5:
        try:
            refined, refined_score = _local_refine(best_points, iu, max_iter=100, tol=1e-12)
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
