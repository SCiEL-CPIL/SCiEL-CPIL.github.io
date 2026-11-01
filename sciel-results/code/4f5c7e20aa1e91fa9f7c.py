"""Packing 50 equal circles in a unit circle using soft-min optimization with multi-stage beta annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the packing score: min of all pair distances/2 and boundary distances."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    return min(min_pair, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints) where constraints are pair distances/2 and boundary distances.
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx] - centers[j_idx]
    pair_dists = np.linalg.norm(diff, axis=1)
    
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # All constraint values
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # Soft min using log-sum-exp for stability
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    S = np.sum(exp_vals)
    log_S = np.log(S)
    obj = -(m - (1.0 / beta) * log_S)
    
    # Weights for soft min
    weights = exp_vals / S
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pair contributions
    pair_weights = weights[:n*(n-1)//2]
    safe_mags = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    unit_vecs = diff / safe_mags[:, np.newaxis]
    
    half_w = pair_weights / 2.0
    np.add.at(grad, i_idx, half_w[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -half_w[:, np.newaxis] * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
    
    return obj, -grad

def _force_directed_relax(centers, start_time, max_iter=3000, target_r=None):
    """
    Force-directed relaxation with adaptive time-stepping.
    """
    current = centers.copy()
    best_score = _score(current)
    
    if target_r is None:
        target_r = best_score * 1.001
    
    k_repel = 80.0
    k_bound = 300.0
    dt = 0.001
    damping = 0.92
    
    vel = np.zeros_like(current)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for iteration in range(max_iter):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        forces = np.zeros_like(current)
        
        # Pair repulsion
        diff = current[i_idx] - current[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        min_dist = 2.0 * target_r
        
        mask = dists < min_dist * 1.1
        if np.any(mask):
            d_vecs = diff[mask]
            d_mags = dists[mask]
            safe_mags = np.where(d_mags > 1e-12, d_mags, 1.0)
            unit_vecs = d_vecs / safe_mags[:, np.newaxis]
            penetration = np.maximum(0, min_dist - d_mags)
            f_mags = k_repel * penetration
            np.add.at(forces, i_idx[mask], f_mags[:, np.newaxis] * unit_vecs)
            np.add.at(forces, j_idx[mask], -f_mags[:, np.newaxis] * unit_vecs)
        
        # Boundary force
        norms = np.linalg.norm(current, axis=1)
        boundary_limit = 1.0 - target_r
        outside = norms > boundary_limit
        if np.any(outside):
            safe_norms = np.where(norms[outside] > 1e-12, norms[outside], 1.0)
            excess = (norms[outside] - boundary_limit)
            f_bound = k_bound * excess
            forces[outside] -= (f_bound / safe_norms)[:, np.newaxis] * current[outside]
        
        # Adaptive time step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-10:
            dt_adaptive = min(dt, 0.005 / max_force)
            dt_adaptive = max(dt_adaptive, 1e-6)
        else:
            dt_adaptive = dt
        
        vel = damping * vel + dt_adaptive * forces
        current = current + dt_adaptive * vel
        
        # Clamp
        norms_after = np.linalg.norm(current, axis=1)
        max_allowed = 1.02
        mask_out = norms_after > max_allowed
        if np.any(mask_out):
            current[mask_out] *= max_allowed / norms_after[mask_out]
            vel[mask_out] *= 0.3
        
        score = _score(current)
        if score > best_score:
            best_score = score
            if best_score > target_r * 0.999:
                target_r = min(best_score * 1.0003, best_score + 0.0005)
        
        if max_force < 1e-9:
            target_r *= 1.0005
            vel *= 0.05
            if target_r > best_score * 1.03:
                break
    
    return current, best_score

def _hex_lattice_init(n, r_est=0.126):
    """Hexagonal lattice initialization."""
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.97:
                points.append([x, y])
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = [[rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)] for _ in range(n * 5)]
    
    points = np.array(points)
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, 0.9)
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _ring_init(n, num_rings=5):
    """Ring-based initialization."""
    centers = [[0.0, 0.0]]
    remaining = n - 1
    ring_radii = np.linspace(0.22, 0.88, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.35
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _random_init(n, seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.87)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers)

def _fibonacci_init(n):
    """Fibonacci-style initialization."""
    centers = [[0.0, 0.0]]
    remaining = n - 1
    golden_angle = 2 * np.pi * (1 - 1 / ((1 + np.sqrt(5)) / 2))
    for i in range(1, remaining + 1):
        radius = 0.15 + 0.72 * np.sqrt(i / remaining)
        angle = i * golden_angle
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers[:n])

def _soft_min_optimize(centers, start_time, beta_sequence=None):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    if beta_sequence is None:
        beta_sequence = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    current = centers.copy()
    best_score = _score(current)
    
    for beta in beta_sequence:
        if time.time() - start_time > _DEADLINE - 8:
            break
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'gtol': 1e-10, 'maxls': 100}
            )
            if res.x is not None:
                optimized = res.x.reshape(N, 2)
                if np.all(np.isfinite(optimized)):
                    r = _score(optimized)
                    if r > best_score:
                        best_score = r
                        current = optimized
        except Exception:
            pass
        
        # Force-directed relaxation after each beta stage
        if time.time() - start_time > _DEADLINE - 6:
            break
        target_r = best_score * 1.001
        fd_result, fd_score = _force_directed_relax(current, start_time, max_iter=1500, target_r=target_r)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                current = fd_result
    
    return current, best_score

def _nelder_mead_polish(centers, maxiter=500):
    """Final polish with Nelder-Mead on the raw nonsmooth objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        return -_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers

def _full_optimize(centers, start_time):
    """Full optimization pipeline."""
    current = centers.copy()
    best_score = _score(current)
    
    # Force-directed pre-conditioning
    if time.time() - start_time < _DEADLINE - 30:
        fd_result, fd_score = _force_directed_relax(current, start_time, max_iter=2000)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                current = fd_result
    
    # Multi-stage soft-min optimization
    if time.time() - start_time < _DEADLINE - 15:
        current, best_score = _soft_min_optimize(current, start_time)
    
    # Final Nelder-Mead polish
    if time.time() - start_time < _DEADLINE - 5:
        nm_result = _nelder_mead_polish(current, maxiter=400)
        r = _score(nm_result)
        if r > best_score:
            best_score = r
            current = nm_result
    
    return current, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _hex_lattice_init(N),
        _ring_init(N),
        _random_init(N, seed=42),
        _random_init(N, seed=123),
        _random_init(N, seed=777),
        _fibonacci_init(N),
        _random_init(N, seed=2024),
    ]
    
    # First pass: force-directed relaxation from each initialization
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 50:
            break
        
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.97
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.97 / norms[mask]
        
        fd_result, fd_score = _force_directed_relax(init, start_time, max_iter=1500)
        if np.all(np.isfinite(fd_result)):
            if fd_score > best_score:
                best_score = fd_score
                best_centers = fd_result.copy()
    
    # Second pass: full optimization on the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 30:
        best_centers, best_score = _full_optimize(best_centers, start_time)
    
    # Third pass: perturb and re-optimize
    if best_centers is not None and time.time() - start_time < _DEADLINE - 20:
        rng = np.random.default_rng(7)
        for attempt in range(4):
            if time.time() - start_time > _DEADLINE - 20:
                break
            
            noise_scale = 0.002 + 0.006 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.97
            if np.any(mask):
                noisy[mask] *= 0.97 / norms[mask]
            
            fd_result, fd_score = _force_directed_relax(noisy, start_time, max_iter=1000)
            if np.all(np.isfinite(fd_result)):
                if fd_score > best_score:
                    best_score = fd_score
                    best_centers = fd_result.copy()
            
            if time.time() - start_time < _DEADLINE - 10:
                opt_result, opt_score = _soft_min_optimize(noisy, start_time, beta_sequence=[200, 500, 1000, 3000])
                if np.all(np.isfinite(opt_result)):
                    if opt_score > best_score:
                        best_score = opt_score
                        best_centers = opt_result.copy()
        
        # Final polish
        if time.time() - start_time < _DEADLINE - 5:
            best_centers, best_score = _full_optimize(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
    
    return best_centers
# EVOLVE-BLOCK-END
