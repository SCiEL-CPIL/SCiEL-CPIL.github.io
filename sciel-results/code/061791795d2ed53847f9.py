"""Improved program for packing 300 circles in a unit circle using soft-min optimization."""
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

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
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
    
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
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

def _force_directed_relax(points, n_steps=100, dt=0.01):
    """
    Simple force-directed relaxation to resolve overlaps and push points inward.
    """
    n = len(points)
    points = points.copy()
    
    iu = np.triu_indices(n, k=1)
    
    for step in range(n_steps):
        forces = np.zeros_like(points)
        
        vecs = points[iu[0]] - points[iu[1]]
        dists = np.linalg.norm(vecs, axis=1)
        mask = dists < 0.1
        if np.any(mask):
            safe_dists = np.where(dists[mask] > 1e-10, dists[mask], 1e-10)
            repulse = (0.1 - dists[mask]) / safe_dists
            force_vecs = repulse[:, np.newaxis] * vecs[mask]
            np.add.at(forces, iu[0][mask], -force_vecs)
            np.add.at(forces, iu[1][mask], force_vecs)
        
        norms = np.linalg.norm(points, axis=1)
        boundary_mask = norms > 0.95
        if np.any(boundary_mask):
            pull_strength = (norms[boundary_mask] - 0.95) * 2.0
            forces[boundary_mask] -= pull_strength[:, np.newaxis] * points[boundary_mask] / norms[boundary_mask, np.newaxis]
        
        points += dt * forces
        
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
    
    return points

def _soft_min_obj_and_grad(x, beta):
    """
    Compute soft-min objective and gradient.
    x is a flat array of 2*N coordinates.
    Returns (objective_value, gradient).
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    vecs = p[i_idx] - p[j_idx]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Soft-min using log-sum-exp trick
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

def _local_optimize(points, beta, maxiter=500, gtol=1e-12, ftol=1e-14, maxls=40):
    """
    Run L-BFGS-B optimization with given parameters.
    """
    try:
        res = minimize(
            lambda x: _soft_min_obj_and_grad(x, beta)[0],
            points.flatten(),
            jac=lambda x: _soft_min_obj_and_grad(x, beta)[1],
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': maxls,
                'disp': False
            }
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
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
    
    # Apply force-directed relaxation to improve initial configuration
    relaxed_points = _force_directed_relax(best_points, n_steps=50, dt=0.005)
    relaxed_score = _compute_score(relaxed_points)
    if relaxed_score > best_score:
        best_points = relaxed_points
        best_score = relaxed_score
    
    # Optimization loop with increasing beta
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 200000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        # Use tighter tolerances for higher betas
        if beta >= 50000:
            maxiter = 300
            gtol = 1e-14
            ftol = 1e-16
            maxls = 60
        elif beta >= 10000:
            maxiter = 400
            gtol = 1e-13
            ftol = 1e-15
            maxls = 50
        elif beta >= 1000:
            maxiter = 500
            gtol = 1e-12
            ftol = 1e-14
            maxls = 40
        elif beta >= 100:
            maxiter = 600
            gtol = 1e-11
            ftol = 1e-13
            maxls = 35
        else:
            maxiter = 500
            gtol = 1e-10
            ftol = 1e-12
            maxls = 30
        
        candidate = _local_optimize(current_points, beta, maxiter, gtol, ftol, maxls)
        
        if time.time() > deadline - 10:
            break
            
        candidate_score = _compute_score(candidate)
        
        if candidate_score > best_score:
            best_points = candidate
            best_score = candidate_score
            current_points = candidate
        else:
            current_points = best_points
    
    # Multi-start perturbation: try small perturbations of best solution
    if time.time() < deadline - 20:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            try:
                perturbed = best_points + np.random.normal(0, 1e-5, (N, 2))
                norms = np.linalg.norm(perturbed, axis=1)
                outside = norms > 1.0
                if np.any(outside):
                    perturbed[outside] /= norms[outside, np.newaxis]
                
                # Quick optimization with medium beta
                candidate = _local_optimize(perturbed, 5000.0, 200, 1e-12, 1e-14, 40)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
            except Exception:
                pass
    
    # Final polish with very high beta if time permits
    if time.time() < deadline - 5:
        try:
            final_beta = 1000000.0
            candidate = _local_optimize(best_points, final_beta, 300, 1e-15, 1e-17, 80)
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
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
