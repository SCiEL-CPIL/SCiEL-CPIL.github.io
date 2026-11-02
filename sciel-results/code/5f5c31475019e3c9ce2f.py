"""
Improved packing of 50 equal circles in a unit circle.
Uses soft-min log-sum-exp with adaptive beta, force-directed pre-conditioning,
and multi-stage L-BFGS-B optimization with perturbation-based restarts.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin for 150s limit
_START_TIME = None


def _compute_score(centers):
    """Compute the packing score for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms >= 1.0):
        return 0.0
    
    r_cont = 1.0 - np.max(norms)
    if r_cont < 0:
        return 0.0
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat


def _force_directed_relax(pts, n_steps=300, target_r=0.125):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with adaptive learning rate
        lr = 0.015 * (1.0 - step / max(n_steps, 1)) + 0.002
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts


def _hex_lattice_init(r_guess=0.125):
    """Generate a hexagonal lattice packing as initial guess."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _random_init(seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(N):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage, beta in enumerate(beta_schedule):
        if _START_TIME is not None and time.time() - _START_TIME > DEADLINE - 10:
            break
        
        def objective(x):
            return _soft_min_obj_and_grad(x, beta)
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                   'gtol': 1e-10, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]
    
    return current_pts


def _polish_subgradient(centers, max_iters=5000, step_size=0.0005):
    """
    Direct approach: Sub-gradient ascent on the minimum distance.
    Identifies the tightest constraint and pushes points apart.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    for it in range(max_iters):
        if _START_TIME is not None and time.time() - _START_TIME > DEADLINE - 5:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        step = step_size
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j: i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                # Move i and j apart
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                # Move i towards center
                current[i] -= vec * step
        
        # Ensure no points go out of bounds (safety check)
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / new_norms[mask]
            
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
            
    return best_centers, best_score


def _perturb_and_retry(centers, n_perturbations=5):
    """
    Perturb the best solution slightly and re-optimize to escape local minima.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    rng = np.random.default_rng(12345)
    
    for p in range(n_perturbations):
        if _START_TIME is not None and time.time() - _START_TIME > DEADLINE - 10:
            break
        
        # Create a perturbed copy
        perturbed = best_centers.copy()
        noise_scale = 0.008 * (1.0 + p * 0.4)
        noise = rng.normal(0, noise_scale, size=perturbed.shape)
        perturbed += noise
        
        # Ensure validity
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            perturbed[mask] *= 0.95 / norms[mask]
            
        # Optimize the perturbed version
        optimized = _soft_min_optimize(perturbed, [100.0, 500.0, 2000.0, 5000.0], max_iter_per_stage=200)
        if np.all(np.isfinite(optimized)):
            score = _compute_score(optimized)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
                
    return best_centers, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _hex_lattice_init(0.125),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        _random_init(1),
        _random_init(42),
        _random_init(99),
        _random_init(123),
        _random_init(777),
    ]
    
    # Beta schedule for multi-stage optimization
    beta_schedule = [50.0, 200.0, 500.0, 2000.0, 5000.0, 10000.0]
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - _START_TIME > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
        
        # Force-directed pre-conditioning
        if time.time() - _START_TIME < DEADLINE - 20:
            preconditioned = _force_directed_relax(init, n_steps=150, target_r=0.125)
        else:
            preconditioned = init
        
        # Multi-stage soft-min optimization
        if time.time() - _START_TIME < DEADLINE - 20:
            optimized = _soft_min_optimize(preconditioned, beta_schedule, max_iter_per_stage=300)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > best_score:
                    best_score = r
                    best_centers = optimized.copy()
        
        # Polish with subgradient
        if time.time() - _START_TIME < DEADLINE - 10 and best_centers is not None:
            polished, polished_score = _polish_subgradient(best_centers, max_iters=3000, step_size=0.0008)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
    
    if best_centers is None:
        return initializations[0]
        
    # Perturbation-retry loop to escape local minima
    if time.time() - _START_TIME < DEADLINE - 15:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, n_perturbations=5)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            
    # Final high-precision polish
    if time.time() - _START_TIME < DEADLINE - 2:
        final_polished, final_score = _polish_subgradient(best_centers, max_iters=10000, step_size=0.0003)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_polished
        
    return best_centers
