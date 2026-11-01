"""Packing 50 equal circles in a unit circle using force-directed relaxation with adaptive time-stepping."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
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

def _force_directed_relax(centers, start_time, max_iter=2000, target_r=None):
    """
    Force-directed relaxation: treat circles as particles with:
    - Repulsive springs for overlapping pairs (stiff springs)
    - Harmonic boundary potential (quadratic penalty for |c_i| > 1 - r)
    
    Uses adaptive time-stepping for stability.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    
    if target_r is None:
        target_r = best_score * 1.001
    
    # Spring constants
    k_repel = 50.0      # repulsion spring constant
    k_bound = 200.0     # boundary spring constant (stiffer to keep inside)
    
    # Adaptive time step
    dt = 0.002
    damping = 0.95
    
    # Velocity
    vel = np.zeros_like(current)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for iteration in range(max_iter):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Pair repulsion: for each pair, if dist < 2*target_r, push apart
        diff = current[i_idx] - current[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        min_dist = 2.0 * target_r
        
        # Only apply force to overlapping or near-overlapping pairs
        mask = dists < min_dist * 1.05
        if np.any(mask):
            d_vecs = diff[mask]
            d_mags = dists[mask]
            safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
            unit_vecs = d_vecs / safe_mags[:, np.newaxis]
            
            # Penetration depth
            penetration = np.maximum(0, min_dist - d_mags)
            
            # Force magnitude (spring: F = k * penetration)
            f_mags = k_repel * penetration
            
            # Apply forces
            np.add.at(forces, i_idx[mask], f_mags[:, np.newaxis] * unit_vecs)
            np.add.at(forces, j_idx[mask], -f_mags[:, np.newaxis] * unit_vecs)
        
        # Boundary force: harmonic well, F = -k_bound * (|c| - (1-target_r)) * c_hat for |c| > 1-target_r
        norms = np.linalg.norm(current, axis=1)
        boundary_limit = 1.0 - target_r
        outside = norms > boundary_limit
        if np.any(outside):
            safe_norms = np.where(norms[outside] > 1e-10, norms[outside], 1.0)
            # Force points inward proportional to how far outside they are
            excess = (norms[outside] - boundary_limit)
            f_bound = k_bound * excess
            forces[outside] -= (f_bound / safe_norms)[:, np.newaxis] * current[outside]
        
        # Adaptive time step based on max force
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-10:
            # Limit step so that max displacement is reasonable
            dt_adaptive = min(dt, 0.01 / max_force)
            dt_adaptive = max(dt_adaptive, 1e-5)
        else:
            dt_adaptive = dt
        
        # Update velocity and position (semi-implicit Euler / symplectic)
        vel = damping * vel + dt_adaptive * forces
        current = current + dt_adaptive * vel
        
        # Clamp to reasonable region
        norms_after = np.linalg.norm(current, axis=1)
        max_allowed = 1.05
        mask_out = norms_after > max_allowed
        if np.any(mask_out):
            current[mask_out] *= max_allowed / norms_after[mask_out]
            vel[mask_out] *= 0.5
        
        # Compute score
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            # Try to increase target_r slightly
            if best_score > target_r * 0.999:
                target_r = min(best_score * 1.0005, best_score + 0.001)
        
        # Check convergence
        if max_force < 1e-8:
            # Try to increase target_r
            target_r *= 1.001
            vel *= 0.1
            if target_r > best_score * 1.05:
                break
    
    return current, best_score

def _get_ring_initialization(n, num_rings=4):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Distribute points across rings with increasing radius
    ring_radii = np.linspace(0.25, 0.9, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.3
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.88)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_fibonacci_initialization(n):
    """Generate Fibonacci-sphere-like initialization projected to disk."""
    centers = []
    centers.append([0.0, 0.0])
    remaining = n - 1
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(1, remaining + 1):
        # Distribute points in rings with Fibonacci angle offset
        ring = i
        count_in_ring = 6  # hexagonal
        idx_in_ring = i - 1
        radius = 0.2 + 0.65 * np.sqrt(idx_in_ring / max(remaining, 1))
        angle = idx_in_ring * 2 * np.pi / golden_ratio
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers[:n])

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _feasible_radius(centers, r, tol=1e-10):
    """Check if all circles of radius r fit inside unit circle without overlap."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound) >= r - tol

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values with numerical stability."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    shifted = v - m
    # Clip to avoid overflow in exp
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
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-12, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    # For each pair k: grad[i] += w_k * unit_vec[k] / 2, grad[j] -= w_k * unit_vec[k] / 2
    # Vectorize using np.add.at
    half_w = pair_weights / 2.0
    np.add.at(grad, i_idx, half_w[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -half_w[:, np.newaxis] * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
    
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

def _binary_search_radius(centers, start_time, r_low, r_high, max_iters=30, tol=1e-10):
    """
    Binary search on radius r. For each candidate r, optimize positions to maximize
    the minimum constraint value. Returns the best feasible radius found.
    """
    best_r = r_low
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        if r_mid <= best_r:
            break
        
        # Try to find positions where all constraints >= r_mid
        # Use soft-min with high beta to approximate the hard constraint
        beta = 500.0 + 100.0 * it  # Increasing sharpness
        
        optimized = _optimize_from_start(best_centers, beta, maxiter=60, gtol=1e-10)
        if not np.all(np.isfinite(optimized)):
            break
        
        score = _compute_score(optimized)
        if score >= r_mid - tol:
            # Feasible, try higher radius
            best_r = max(best_r, score)
            best_centers = optimized.copy()
            r_low = r_mid
        else:
            # Not feasible, try lower radius
            r_high = r_mid
    
    return best_r, best_centers

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

def _adaptive_radius_subgradient(centers, start_time, max_iter=500, tol=1e-8):
    """
    Custom differentiable subgradient method that adaptively increases radius r.
    """
    current = centers.copy()
    best_r = _compute_score(current)
    r = best_r * 0.99
    
    step_size = 0.01
    patience = 0
    max_patience = 30
    
    for iteration in range(max_iter):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        boundary_dists = 1.0 - norms
        
        pair_min = np.min(dists)
        bound_min = np.min(boundary_dists)
        current_r = min(pair_min / 2.0, bound_min)
        
        target_r = r + step_size * 0.1
        
        grad = np.zeros_like(current)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        violations = np.maximum(0, 2 * target_r - pair_dists)
        
        if np.any(violations > 1e-10):
            d_vecs = current[i_idx] - current[j_idx]
            safe_mags = np.where(pair_dists > 1e-10, pair_dists, 1.0)
            unit_vecs = d_vecs / safe_mags[:, np.newaxis]
            weights = violations / (2 * target_r + 1e-10)
            np.add.at(grad, i_idx, weights[:, np.newaxis] * unit_vecs)
            np.add.at(grad, j_idx, -weights[:, np.newaxis] * unit_vecs)
        
        boundary_violations = np.maximum(0, norms - (1.0 - target_r))
        if np.any(boundary_violations > 1e-10):
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            grad -= (boundary_violations / safe_norms)[:, np.newaxis] * current
        
        total_violation = np.sum(violations) + np.sum(boundary_violations)
        if total_violation < 1e-6:
            r = target_r
            patience = 0
            if iteration % 20 == 0:
                rng = np.random.default_rng(iteration)
                current += rng.normal(0, 0.002, current.shape)
        else:
            current = current + step_size * grad
            norms_after = np.linalg.norm(current, axis=1)
            max_allowed = 1.0 - 0.01
            mask = norms_after > max_allowed
            if np.any(mask):
                current[mask] *= max_allowed / norms_after[mask]
            patience += 1
            if patience > max_patience:
                step_size *= 0.7
                patience = 0
                if step_size < 1e-5:
                    break
        
        current_score = _compute_score(current)
        if current_score > best_r:
            best_r = current_score
            r = best_r * 0.995
    
    return current, best_r

def _polish(centers, start_time):
    """Progressively polish the solution with force-directed relaxation, L-BFGS-B, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Force-directed relaxation first (good for escaping local minima)
    if time.time() - start_time < _DEADLINE - 30:
        fd_result, fd_score = _force_directed_relax(current, start_time, max_iter=1000)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                current = fd_result
    
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
    
    # Force-directed again with tighter target
    if time.time() - start_time < _DEADLINE - 15:
        target_r = best_score * 1.002
        fd_result, fd_score = _force_directed_relax(current, start_time, max_iter=800, target_r=target_r)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                current = fd_result
    
    # Binary search on radius to push higher
    if time.time() - start_time < _DEADLINE - 10:
        r_low = best_score
        r_high = best_score * 1.02
        bs_r, bs_centers = _binary_search_radius(current, start_time, r_low, r_high, max_iters=20)
        if bs_r > best_score:
            best_score = bs_r
            current = bs_centers
    
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
        _get_fibonacci_initialization(N),
    ]
    
    # First pass: force-directed relaxation from each initialization (shorter budget)
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 50:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        # Force-directed relaxation (fast, good for escaping local minima)
        fd_result, fd_score = _force_directed_relax(init, start_time, max_iter=600)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                best_centers = fd_result.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: try to improve with perturbations of the best (fewer attempts, more time each)
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
            
            # Force-directed relaxation from perturbed state
            fd_result, fd_score = _force_directed_relax(noisy, start_time, max_iter=500)
            if np.all(np.isfinite(fd_result)):
                if fd_score > best_score:
                    best_score = fd_score
                    best_centers = fd_result.copy()
            
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
