"""Improved program for packing 300 circles in a unit circle using regularized soft-min optimization."""
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

def _force_directed_relax(points, iterations=80, eta=0.008):
    """
    Perform force-directed relaxation to spread out points.
    This acts as a pre-conditioner for the main optimizer.
    """
    p = points.copy()
    n = len(p)
    
    for it in range(iterations):
        forces = np.zeros_like(p)
        
        # Pairwise repulsive forces: minimize sum 1/d^2
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_d4 = 1.0 / (dists_sq ** 2)
        np.fill_diagonal(inv_d4, 0.0)
        
        pair_forces = 2.0 * inv_d4[:, :, np.newaxis] * diff
        forces += np.sum(pair_forces, axis=1)
        
        # Boundary repulsion
        norms = np.linalg.norm(p, axis=1)
        mask = norms > 0.85
        if np.any(mask):
            mag = (norms[mask] - 0.85) * 15.0
            forces[mask] -= (mag[:, np.newaxis] / norms[mask, np.newaxis]) * p[mask]
            
        p += eta * forces
        
        # Clip to unit circle
        norms = np.linalg.norm(p, axis=1)
        outside = norms > 0.999
        if np.any(outside):
            p[outside] /= norms[outside, np.newaxis] * 1.005

    return p

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    max_k = int(1.0 / (s * 0.5)) + 3
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.92
        max_k = int(1.0 / (s * 0.5)) + 4
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
            if diff == 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts)-1, -1, -1):
            rem = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= rem
            diff += rem
            if diff == 0:
                break
    
    ring_counts = [c for c in ring_counts if c > 0]
    num_rings = len(ring_counts)
    
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    for k, count in enumerate(ring_counts):
        if k == 0:
            points.append([0.0, 0.0])
        else:
            r_k = k * s * 0.95
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.pi/count
            for angle in angles:
                points.append([r_k * np.cos(angle), r_k * np.sin(angle)])
    
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def _objective_and_grad(x, beta, lam, iu, i_idx, j_idx):
    """
    Compute objective value and gradient for the soft-min packing objective.
    Uses pre-computed index arrays for speed.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    # Upper triangle indices
    pair_dists_sq = dists_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Soft-min computation using log-sum-exp trick
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists
    
    max_z = max(np.max(z_pairs), np.max(z_bounds))
    
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    log_sum_exp = max_z + np.log(sum_exp)
    
    val_softmin = log_sum_exp / beta
    
    # Weights
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp
    
    # Gradient accumulation
    grad = np.zeros_like(p)
    
    # Pair part: grad of -soft_min
    # c_ij = 0.5 * |p_i - p_j|
    # grad_{p_i} c_ij = 0.5 * (p_i - p_j) / |p_i - p_j|
    # grad_{p_j} c_ij = -0.5 * (p_i - p_j) / |p_i - p_j|
    
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
    
    # Boundary part
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad = -p / safe_norms[:, np.newaxis]
    grad += -w_bounds[:, np.newaxis] * bound_grad
    
    # Repulsive energy part
    if lam > 0:
        epsilon = 1e-4
        repulsive_term = 1.0 / (pair_dists_sq + epsilon)
        val_repulsive = np.sum(repulsive_term)
        
        rep_coeff = -2.0 * lam / (pair_dists_sq + epsilon)**2
        
        np.add.at(grad, i_idx, rep_coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -rep_coeff[:, np.newaxis] * vecs)
        
        val_total = val_softmin + lam * val_repulsive
    else:
        val_total = val_softmin
    
    return val_total, grad

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    np.random.seed(42)
    
    # Pre-compute upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    # 1. Initialization
    inits = [
        _hex_lattice_init(N),
        _ring_init(N),
    ]
    
    random_points = np.random.rand(N, 2) * 2 - 1
    norms = np.linalg.norm(random_points, axis=1)
    mask = norms > 1.0
    random_points[mask] = random_points[mask] / norms[mask, np.newaxis] * 0.95
    inits.append(random_points)
    
    best_points = inits[0].copy()
    best_score = _compute_score(inits[0])
    
    for init in inits[1:]:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # 2. Force-directed relaxation as pre-conditioner
    current_points = _force_directed_relax(best_points, iterations=60, eta=0.006)
    current_score = _compute_score(current_points)
    if current_score > best_score:
        best_score = current_score
        best_points = current_points
    
    # 3. Optimization with smooth schedule
    schedule = [
        (5.0, 0.8),
        (15.0, 0.6),
        (30.0, 0.4),
        (60.0, 0.3),
        (120.0, 0.2),
        (250.0, 0.1),
        (500.0, 0.05),
        (1000.0, 0.02),
        (2000.0, 0.0),
        (4000.0, 0.0),
        (8000.0, 0.0),
    ]
    
    for beta, lam in schedule:
        if time.time() > deadline - 12:
            break
            
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta, lam, iu, i_idx, j_idx)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta, lam, iu, i_idx, j_idx)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-9,
                    'ftol': 1e-11,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 6:
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

    # 4. Final high-precision polish
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, 8000.0, 0.0, iu, i_idx, j_idx)[0],
                best_points.flatten(),
                jac=lambda x: _objective_and_grad(x, 8000.0, 0.0, iu, i_idx, j_idx)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 150,
                    'gtol': 1e-11,
                    'ftol': 1e-13,
                    'maxls': 15,
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

    # 5. Ultra-precise final pass
    if time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, 15000.0, 0.0, iu, i_idx, j_idx)[0],
                best_points.flatten(),
                jac=lambda x: _objective_and_grad(x, 15000.0, 0.0, iu, i_idx, j_idx)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 12,
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
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
