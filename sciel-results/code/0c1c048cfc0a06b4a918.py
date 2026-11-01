"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # 20s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n*2,) array of centers
    Returns (objective_value, gradient)
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, -grad


def _optimize_lbfgs(centers, beta, maxiter=500, gtol=1e-8):
    """Optimize using L-BFGS-B with analytical gradient."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization matching the known optimal structure.
    The best known packing has rings: 1, 6, 12, 18, 13 = 50
    """
    rng = np.random.default_rng(seed)
    
    # Known optimal structure: 1 center + 4 rings
    rings = [1, 6, 12, 18, 13]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    # Optimal ring radii approximately
    ring_radii = [0.0, 0.2516, 0.5032, 0.7548, 0.9564]
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_radii[min(ring_idx, len(ring_radii) - 1)]
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                # Small jitter to break perfect symmetry
                r_jitter = rng.normal(0, 0.003)
                a_jitter = rng.normal(0, 0.003)
                x += r_jitter * np.cos(angle) + a_jitter * radius * np.sin(angle)
                y += r_jitter * np.sin(angle) - a_jitter * radius * np.cos(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Ensure all points are inside
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.98
    if np.any(mask):
        points[mask] *= (0.98 / norms[mask])[:, np.newaxis]
    
    return points


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization as fallback."""
    rng = np.random.default_rng(42 + seed_offset)
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _force_directed_relax(centers, start_time, max_steps=1000):
    """
    Force-directed relaxation to escape local minima.
    """
    current = centers.copy()
    n = len(current)
    best_score = _score(current)
    best_positions = current.copy()
    stuck = 0
    
    for step in range(max_steps):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        forces = np.zeros_like(current)
        
        # Repulsive forces between all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = current[i_idx, :] - current[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Strong repulsion when close
        repulsion_strength = 1.0 / (safe_dists ** 2 + 1e-6)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        weighted_forces = repulsion_strength[:, np.newaxis] * unit_vecs
        
        np.add.at(forces, i_idx, weighted_forces)
        np.add.at(forces, j_idx, -weighted_forces)
        
        # Boundary constraint: push inward if too close to edge
        norms = np.linalg.norm(current, axis=1)
        boundary_force = np.where(norms > 0.9, (norms - 0.9) ** 2, 0.0)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        forces -= boundary_force[:, np.newaxis] * current / safe_norms[:, np.newaxis]
        
        # Adaptive time step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        dt = 0.002 / (1.0 + max_force * 5)
        
        current += dt * forces
        
        # Keep inside unit circle
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            current[mask] *= (1.0 / new_norms[mask])[:, np.newaxis]
        
        new_score = _score(current)
        if new_score > best_score:
            best_score = new_score
            best_positions = current.copy()
            stuck = 0
        else:
            stuck += 1
            if stuck > 50:
                break
    
    return best_positions


def _multi_beta_optimize(centers, start_time):
    """
    Multi-stage optimization with increasing beta for precision.
    """
    current = centers.copy()
    best_score = _score(current)
    best_positions = current.copy()
    
    # Stages: (beta, maxiter)
    stages = [
        (50, 200),
        (200, 300),
        (1000, 400),
        (5000, 500),
        (20000, 600),
        (50000, 800),
    ]
    
    for beta, maxiter in stages:
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        optimized = _optimize_lbfgs(current, beta, maxiter=maxiter)
        opt_score = _score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_positions = optimized.copy()
            current = optimized.copy()
    
    return best_positions, best_score


def _nelder_mead_polish(pts0, start_time):
    """Final high-precision polish using Nelder-Mead on the exact objective."""
    
    def exact_obj(x):
        pts = x.reshape(N, 2)
        dists = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dists > 1.0):
            return 1e6 + np.sum(dists[dists > 1.0] - 1.0)
        r_cont = 1.0 - np.max(dists)
        if N < 2:
            return -r_cont
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, np.inf)
        r_pair = 0.5 * np.min(dist_mat)
        return -min(r_cont, r_pair)
    
    def callback_polish(xk):
        return time.time() - start_time > _DEADLINE - 5.0
    
    try:
        res = minimize(
            exact_obj,
            pts0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            callback=callback_polish
        )
        best_pts = res.x.reshape(N, 2)
        
        # Project back to unit disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis]
        
        return best_pts
    except Exception:
        return pts0


def construct():
    start_time = time.time()
    
    # Primary: ring-based initialization matching known optimal structure
    pts_ring = _get_ring_init(N, seed=42)
    best_pts = pts_ring
    best_score = _score(pts_ring)
    
    # Try multiple ring seeds to find best initialization
    for seed in [0, 1, 2, 3, 7, 13, 42, 99, 123, 256]:
        if time.time() - start_time > _DEADLINE - 60:
            break
        pts_try = _get_ring_init(N, seed=seed)
        s = _score(pts_try)
        if s > best_score:
            best_score = s
            best_pts = pts_try
    
    # Fallback: hexagonal lattice if ring init is poor
    pts_hex = _get_hex_lattice(N, seed_offset=0)
    hex_score = _score(pts_hex)
    if hex_score > best_score:
        best_score = hex_score
        best_pts = pts_hex
    
    # Phase 1: Force-directed relaxation to escape bad local minima
    if time.time() - start_time < _DEADLINE - 40:
        relaxed = _force_directed_relax(best_pts, start_time, max_steps=500)
        rel_score = _score(relaxed)
        if rel_score > best_score:
            best_score = rel_score
            best_pts = relaxed
    
    # Phase 2: Multi-beta L-BFGS-B optimization
    if time.time() - start_time < _DEADLINE - 20:
        opt_pts, opt_score = _multi_beta_optimize(best_pts, start_time)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Phase 3: Another force-directed relaxation after optimization
    if time.time() - start_time < _DEADLINE - 15:
        relaxed2 = _force_directed_relax(best_pts, start_time, max_steps=300)
        rel2_score = _score(relaxed2)
        if rel2_score > best_score:
            best_score = rel2_score
            best_pts = relaxed2
    
    # Phase 4: Final L-BFGS-B polish with high beta
    if time.time() - start_time < _DEADLINE - 10:
        final_opt = _optimize_lbfgs(best_pts, 100000, maxiter=500, gtol=1e-10)
        final_score = _score(final_opt)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_opt
    
    # Phase 5: Nelder-Mead for final precision
    if time.time() - start_time < _DEADLINE - 5:
        polish_pts = _nelder_mead_polish(best_pts, start_time)
        polish_score = _score(polish_pts)
        if polish_score > best_score:
            best_score = polish_score
            best_pts = polish_pts
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts_ring
    
    return best_pts

# EVOLVE-BLOCK-END
