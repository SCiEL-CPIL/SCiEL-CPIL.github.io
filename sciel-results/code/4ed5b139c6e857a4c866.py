"""Packing 50 equal circles in a unit circle using PBD + soft-min gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the maximum radius r for which circles centered at 'centers' are disjoint and inside the unit circle."""
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    norms = np.linalg.norm(centers, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values with numerical stability."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    shifted = v - m
    shifted = np.clip(shifted, 0, 50.0 / beta)
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return m - (1.0 / beta) * log_sum

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    obj = -_soft_min(vals, beta)
    
    # Compute weights with numerical stability
    m = np.min(vals)
    shifted = vals - m
    shifted = np.clip(shifted, 0, 50.0 / beta)
    exp_vals = np.exp(-beta * shifted)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    mask = pair_weights > 1e-12
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = pair_dists[mask]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
        contrib = w_m[:, np.newaxis] * 0.5 * vecs
        np.add.at(grad, i_m, contrib)
        np.add.at(grad, j_m, -contrib)
    
    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
        vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
        contrib_b = w_b[:, np.newaxis] * vecs_b
        np.add.at(grad, idx_b, -contrib_b)
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-12}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _pbd_relax(centers, start_time, max_iter=500, target_r=None):
    """
    Position-Based Dynamics relaxation with hard constraint projection.
    Each iteration:
    1. Project centers to stay inside unit disk (radius 1 - target_r)
    2. Resolve pairwise overlaps by moving apart along connecting vectors
    This ensures hard feasibility at every step.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    
    if target_r is None:
        target_r = best_score * 1.001
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for iteration in range(max_iter):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        # Step 1: Resolve pairwise overlaps (iterate a few times for convergence)
        for _ in range(3):
            diff = current[i_idx] - current[j_idx]
            dists = np.linalg.norm(diff, axis=1)
            min_dist = 2.0 * target_r
            
            # Find overlapping pairs
            mask = dists < min_dist - 1e-12
            if not np.any(mask):
                break
            
            d_vecs = diff[mask]
            d_mags = dists[mask]
            safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
            unit_vecs = d_vecs / safe_mags[:, np.newaxis]
            
            # Each circle moves half the overlap
            overlap = (min_dist - d_mags) / 2.0
            np.add.at(current, i_idx[mask], overlap[:, np.newaxis] * unit_vecs)
            np.add.at(current, j_idx[mask], -overlap[:, np.newaxis] * unit_vecs)
        
        # Step 2: Project to unit disk boundary
        norms = np.linalg.norm(current, axis=1)
        boundary_limit = 1.0 - target_r
        outside = norms > boundary_limit
        if np.any(outside):
            current[outside] *= boundary_limit / norms[outside]
        
        # Compute score and update target
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            if best_score > target_r * 0.999:
                target_r = min(best_score * 1.0005, best_score + 0.001)
        
        # Check convergence
        diff = current[i_idx] - current[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        min_dist = 2.0 * target_r
        max_violation = np.max(np.maximum(0, min_dist - dists))
        norms_after = np.linalg.norm(current, axis=1)
        max_boundary_violation = np.max(np.maximum(0, norms_after - boundary_limit))
        
        if max_violation < 1e-10 and max_boundary_violation < 1e-10:
            target_r *= 1.0005
            if target_r > best_score * 1.05:
                break
    
    return current, best_score

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])
                
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, num_rings=5):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Approximate optimal distribution: center + rings of 6, 12, 12, 9, 10
    ring_configs = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
    ]
    
    count = 1
    for count_in_ring, radius in ring_configs:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(10000):
        if len(centers) >= n:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > 0.95**2:
            continue
        if centers:
            pts = np.array(centers)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            if np.min(dists) < 0.15:
                continue
        centers.append([x, y])
    
    while len(centers) < n:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        centers.append([x, y])
        
    return np.array(centers[:n])

def _nelder_mead_polish(centers, maxiter=200, tol=1e-10):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        return -_compute_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _high_precision_polish(centers, start_time):
    """
    Final high-precision polish: multiple rounds of L-BFGS-B with very high beta
    and tight tolerances, followed by Nelder-Mead on the nonsmooth objective.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Multiple L-BFGS-B passes with increasing beta and tight tolerances
    betas = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=200, gtol=1e-12)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # Nelder-Mead polish with very tight tolerances
    if time.time() - start_time < _DEADLINE - 3:
        nm_result = _nelder_mead_polish(current, maxiter=500)
        if np.all(np.isfinite(nm_result)):
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    # One final L-BFGS-B pass with the highest beta
    if time.time() - start_time < _DEADLINE - 1:
        final = _optimize_from_start(current, 20000.0, maxiter=100, gtol=1e-14)
        if np.all(np.isfinite(final)):
            r = _compute_score(final)
            if r > best_score:
                best_score = r
                current = final
    
    return current, best_score

def _polish(centers, start_time):
    """Progressively polish the solution with PBD, L-BFGS-B, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # PBD relaxation first (fast, good for escaping local minima with hard constraints)
    if time.time() - start_time < _DEADLINE - 30:
        pbd_result, pbd_score = _pbd_relax(current, start_time, max_iter=800)
        if np.all(np.isfinite(pbd_result)):
            if pbd_score > best_score:
                best_score = pbd_score
                current = pbd_result
    
    # L-BFGS-B with increasing beta
    betas = [100.0, 300.0, 800.0, 2000.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # PBD again with tighter target
    if time.time() - start_time < _DEADLINE - 15:
        target_r = best_score * 1.002
        pbd_result, pbd_score = _pbd_relax(current, start_time, max_iter=600, target_r=target_r)
        if np.all(np.isfinite(pbd_result)):
            if pbd_score > best_score:
                best_score = pbd_score
                current = pbd_result
    
    # High-precision final polish
    if time.time() - start_time < _DEADLINE - 5:
        current, best_score = _high_precision_polish(current, start_time)
    
    return current, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
    ]
    
    # First pass: PBD relaxation from each initialization (shorter budget)
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 50:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        # PBD relaxation (fast, good for escaping local minima with hard constraints)
        pbd_result, pbd_score = _pbd_relax(init, start_time, max_iter=500)
        if np.all(np.isfinite(pbd_result)):
            if pbd_score > best_score:
                best_score = pbd_score
                best_centers = pbd_result.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: try to improve with perturbations of the best
    if best_centers is not None and time.time() - start_time < _DEADLINE - 20:
        rng = np.random.default_rng(7)
        for attempt in range(4):
            if time.time() - start_time > _DEADLINE - 20:
                break
            
            noise_scale = 0.003 + 0.008 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                noisy[mask] *= 0.98 / norms[mask]
            
            # PBD relaxation from perturbed state
            pbd_result, pbd_score = _pbd_relax(noisy, start_time, max_iter=400)
            if np.all(np.isfinite(pbd_result)):
                if pbd_score > best_score:
                    best_score = pbd_score
                    best_centers = pbd_result.copy()
            
            # Also try L-BFGS-B
            if time.time() - start_time < _DEADLINE - 15:
                optimized = _optimize_from_start(noisy, beta=500.0, maxiter=80, gtol=1e-10)
                if np.all(np.isfinite(optimized)):
                    r = _compute_score(optimized)
                    if r > best_score:
                        best_score = r
                        best_centers = optimized.copy()
        
        # Final high-precision polish
        if time.time() - start_time < _DEADLINE - 5:
            best_centers, best_score = _high_precision_polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
