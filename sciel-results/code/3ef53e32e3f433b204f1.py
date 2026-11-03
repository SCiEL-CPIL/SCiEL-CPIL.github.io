import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0

# Ring structure: (count, base_radius_fraction)
# Based on known optimal packing structure
RING_STRUCTURE = [
    (1, 0.0),      # center
    (6, 0.30),     # inner ring
    (12, 0.55),    # middle ring
    (18, 0.80),    # outer ring
    (13, 0.95),    # outermost ring
]
# Total: 1+6+12+18+13 = 50

RING_COUNTS = [r[0] for r in RING_STRUCTURE]
RING_BASES = [r[1] for r in RING_STRUCTURE]
N_RINGS = len(RING_STRUCTURE)


def _ring_to_points(radii, angles):
    """Convert ring radii and angles to point coordinates.
    
    radii: shape (N_RINGS,)
    angles: shape (N,) - one angle per point
    Returns: shape (N, 2)
    """
    points = np.zeros((N, 2))
    start = 0
    for ring_idx in range(N_RINGS):
        count = RING_COUNTS[ring_idx]
        r = radii[ring_idx]
        a = angles[start:start + count]
        points[start:start + count, 0] = r * np.cos(a)
        points[start:start + count, 1] = r * np.sin(a)
        start += count
    return points


def _score(points):
    """Compute the packing score (largest common radius)."""
    points = np.asarray(points).reshape(N, 2)
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    r = min(np.min(boundary), np.min(pair_dists))
    return r


def _softmin_logsumexp(values, beta):
    """Numerically stable soft-min."""
    m = np.min(values)
    if not np.isfinite(m):
        return m
    shifted = -beta * (values - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / beta


def _smooth_objective_ring(radii, angles, beta):
    """Smooth objective for ring parameterization."""
    points = _ring_to_points(radii, angles)
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([pair_dists, boundary])
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _smooth_grad_free(x_flat, beta):
    """Gradient of smooth objective for free coordinates."""
    points = x_flat.reshape(N, 2)
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    all_vals = np.concatenate([pair_dists, boundary])
    m = np.min(all_vals)
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -500, 500)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros((N, 2))
    
    for idx in range(len(pair_dists)):
        w = weights[idx]
        if w < 1e-12:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d_vec = points[i] - points[j]
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-12:
            continue
        unit = d_vec / d_norm
        grad[i] += 0.5 * w * unit
        grad[j] -= 0.5 * w * unit
    
    for i in range(N):
        w = weights[len(pair_dists) + i]
        if w < 1e-12:
            continue
        if norms[i] < 1e-12:
            continue
        grad[i] -= w * points[i] / norms[i]
    
    return grad.flatten()


def _optimize_free(x0, time_limit=10.0, beta_start=100.0, beta_end=5000.0):
    """Optimize free coordinates with progressive beta annealing."""
    t_start = time.time()
    x = x0.flatten().copy()
    n_betas = 8
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    best_x = x.copy()
    best_score = _score(best_x.reshape(N, 2))
    
    for b in betas:
        if time.time() - t_start > time_limit:
            break
        
        def obj(x_flat):
            points = x_flat.reshape(N, 2)
            norms = np.hypot(points[:, 0], points[:, 1])
            boundary = 1.0 - norms
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            iu = np.triu_indices(N, 1)
            pair_dists = dists[iu] / 2.0
            all_vals = np.concatenate([pair_dists, boundary])
            return -_softmin_logsumexp(all_vals, b)
        
        try:
            res = minimize(
                obj, x,
                jac=lambda xf: _smooth_grad_free(xf, b),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < obj(x):
                x = res.x
                s = _score(x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_score


def _optimize_ring(radii0, angles0, time_limit=15.0):
    """Optimize ring parameters: first DE on radii, then L-BFGS on angles."""
    t_start = time.time()
    best_points = _ring_to_points(radii0, angles0)
    best_score = _score(best_points)
    
    # Phase 1: DE on ring radii (global search over 5 dimensions)
    if time.time() - t_start < time_limit * 0.5:
        bounds = []
        for i in range(N_RINGS):
            if RING_COUNTS[i] == 1:
                bounds.append((0.0, 0.1))
            else:
                base = RING_BASES[i]
                bounds.append((base * 0.6, base * 1.3))
        
        def de_obj(radii):
            points = _ring_to_points(radii, angles0)
            norms = np.hypot(points[:, 0], points[:, 1])
            boundary = 1.0 - norms
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            iu = np.triu_indices(N, 1)
            pair_dists = dists[iu] / 2.0
            all_vals = np.concatenate([pair_dists, boundary])
            return -_softmin_logsumexp(all_vals, 500.0)
        
        try:
            de_res = differential_evolution(
                de_obj, bounds,
                seed=42, maxiter=100, tol=1e-10,
                popsize=15, polish=False
            )
            if de_res.fun < de_obj(radii0):
                radii0 = de_res.x
                best_points = _ring_to_points(radii0, angles0)
                best_score = _score(best_points)
        except Exception:
            pass
    
    # Phase 2: L-BFGS on angles (local refinement)
    if time.time() - t_start < time_limit:
        angles = angles0.copy()
        
        for beta in [200.0, 1000.0, 3000.0]:
            if time.time() - t_start > time_limit:
                break
            
            def angle_obj(a_flat):
                points = _ring_to_points(radii0, a_flat)
                norms = np.hypot(points[:, 0], points[:, 1])
                boundary = 1.0 - norms
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                iu = np.triu_indices(N, 1)
                pair_dists = dists[iu] / 2.0
                all_vals = np.concatenate([pair_dists, boundary])
                return -_softmin_logsumexp(all_vals, beta)
            
            try:
                res = minimize(
                    angle_obj, angles.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15}
                )
                if res.fun < angle_obj(angles):
                    angles = res.x
                    pts = _ring_to_points(radii0, angles)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
            except Exception:
                pass
    
    return best_points, best_score, radii0, angles


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.2, 20):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _ring_init():
    """Generate ring-based initial configuration."""
    angles = np.zeros(N)
    start = 0
    for ring_idx in range(N_RINGS):
        count = RING_COUNTS[ring_idx]
        for k in range(count):
            angles[start + k] = 2 * np.pi * k / count
        start += count
    radii = np.array(RING_BASES)
    return radii, angles


def construct():
    t_start = time.time()
    
    # 1. Initial configurations
    init_hex = _hexagonal_init(N)
    radii0, angles0 = _ring_init()
    init_ring = _ring_to_points(radii0, angles0)
    
    best_points = init_hex.copy()
    best_score = max(_score(init_hex), _score(init_ring))
    if _score(init_ring) > _score(init_hex):
        best_points = init_ring.copy()
    
    # 2. Ring-based optimization (primary method)
    if time.time() - t_start < DEADLINE - 30:
        tlim = min(25.0, DEADLINE - (time.time() - t_start) - 30)
        pts, sc, radii0, angles0 = _optimize_ring(radii0, angles0, time_limit=tlim)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # 3. Free-coordinate polish from best
    if time.time() - t_start < DEADLINE - 20:
        tlim = min(15.0, DEADLINE - (time.time() - t_start) - 20)
        c_opt, r_opt = _optimize_free(best_points, time_limit=tlim, beta_start=200.0, beta_end=5000.0)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 4. Multi-start perturbations
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 10 and n_restarts < 4:
        scale = 0.04 / (1.0 + 0.5 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * scale
        tlim = min(12.0, DEADLINE - (time.time() - t_start) - 10)
        c_opt, r_opt = _optimize_free(x0, time_limit=tlim, beta_start=300.0, beta_end=8000.0)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        n_restarts += 1
    
    # 5. Final high-beta polish
    if time.time() - t_start < DEADLINE - 3:
        tlim = min(5.0, DEADLINE - (time.time() - t_start) - 3)
        c_opt, r_opt = _optimize_free(best_points, time_limit=tlim, beta_start=2000.0, beta_end=15000.0)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1.0:
            scale = (1.0 - 1e-10) / r2[i]
            c_final[i] *= scale
    
    return c_final
