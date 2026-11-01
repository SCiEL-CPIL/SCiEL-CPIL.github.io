"""Optimize 50 circle centres in the unit disk via structured initialization + multi-method refinement."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0


def _score(points):
    """Compute the packing radius (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    Minimizes -r where r is the packing radius, using a smooth approximation.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distances to boundary (1 - |p_i|)
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine all constraints into a single vector of "margins"
    pair_margins = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_margin_vec = pair_margins[iu]
    
    # All margins
    all_margins = np.concatenate([pair_margin_vec, boundary_dists])
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta * m))) / beta
    neg_beta_m = -beta * all_margins
    max_val = np.max(neg_beta_m)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_m - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    # Objective: minimize -soft_min_val (i.e., maximize soft_min_val)
    obj = -soft_min_val
    
    # Gradient with respect to x
    weights = np.exp(neg_beta_m - max_val)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(x)
    
    # Pair gradients
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d > 1e-12:
            w = weights[idx]
            diff_ij = pts[i] - pts[j]
            grad[2*i:2*i+2] += w * diff_ij / d
            grad[2*j:2*j+2] -= w * diff_ij / d
    
    # Boundary gradients
    for i in range(N):
        r = radii[i]
        if r > 1e-12:
            w = weights[len(pair_margin_vec) + i]
            grad[2*i:2*i+2] += w * pts[i] / r
    
    return obj, grad


def _hex_lattice_init():
    """Generate hexagonal lattice initial configuration."""
    best_pts = None
    best_score = -1
    for trial in range(80):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + 0.005 * trial)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= N:
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(radii, kind="stable")[:N]
            candidate = pts[idx]
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return _random_init(42)


def _ring_init():
    """Ring-based initialization with multiple rings."""
    # Use a structured ring layout
    n_rings = 5
    pts = []
    
    # Center point
    pts.append([0.0, 0.0])
    
    # Distribute remaining points in rings
    remaining = N - 1
    pts_per_ring = [remaining * w for w in [0.1, 0.2, 0.25, 0.25, 0.2]]
    pts_per_ring = [int(p) for p in pts_per_ring]
    # Adjust to sum to remaining
    diff = remaining - sum(pts_per_ring)
    pts_per_ring[-1] += diff
    
    for ring_idx, count in enumerate(pts_per_ring):
        if count <= 0:
            continue
        radius = 0.2 + 0.15 * ring_idx
        angles = 2 * np.pi * np.arange(count) / count + ring_idx * 0.1
        for a in angles:
            pts.append([radius * np.cos(a), radius * np.sin(a)])
    
    pts = np.array(pts[:N])
    return pts


def _random_init(seed=None):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 0.85 ** 2:
                pts[i] = [x, y]
                break
    return pts


def _golden_angle_init():
    """Golden angle spiral initialization."""
    pts = np.zeros((N, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.95
        theta = i * golden_angle
        pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    return pts


def _perturb(pts, sigma, seed=None):
    """Perturb points with Gaussian noise and clamp to disk."""
    rng = np.random.default_rng(seed)
    new_pts = pts + rng.normal(0, sigma, size=pts.shape)
    r = np.sqrt(np.sum(new_pts ** 2, axis=1))
    mask = r > 0.95
    if np.any(mask):
        scale = 0.95 / r[mask]
        new_pts[mask] *= scale[:, None]
    return new_pts


def _optimize_lbfgs(x0, beta, maxiter=2000, gtol=1e-8, deadline=None):
    """Run L-BFGS-B optimization for a given beta."""
    def obj_and_grad(x):
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': 1e-12,
                'maxfun': maxiter * 2
            }
        )
        return result.x
    except Exception:
        return x0


def _force_directed_relax(pts, iterations=500, dt=0.01):
    """
    Force-directed relaxation to spread points apart.
    Points repel each other and are attracted to stay within the disk.
    """
    pts = pts.copy()
    
    for it in range(iterations):
        forces = np.zeros_like(pts)
        
        # Repulsion between all pairs
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # Repulsive force proportional to 1/dist
                    f = diff / dist ** 2
                    forces[i] += f
                    forces[j] -= f
        
        # Attraction toward origin (to keep within disk)
        for i in range(N):
            r = np.linalg.norm(pts[i])
            if r > 0.5:
                forces[i] -= 0.1 * pts[i] / r
        
        # Update positions
        pts += dt * forces
        
        # Clamp to disk
        r = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = r > 0.95
        if np.any(mask):
            scale = 0.95 / r[mask]
            pts[mask] *= scale[:, None]
    
    return pts


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    
    # Generate multiple diverse initial configurations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    inits.append(_golden_angle_init())
    inits.append(_random_init(seed=42))
    inits.append(_random_init(seed=123))
    
    # Apply force-directed relaxation to improve initial configs
    relaxed_inits = []
    for init in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        relaxed = _force_directed_relax(init, iterations=200, dt=0.02)
        relaxed_inits.append(relaxed)
    
    best_score = -1
    best_pts = None
    
    # Evaluate all initializations
    for init in inits + relaxed_inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    # Phase 1: Global search with moderate beta
    betas_phase1 = [50, 100, 200, 500]
    for beta in betas_phase1:
        if time.time() - start_time > DEADLINE - 30:
            break
        x_opt = _optimize_lbfgs(best_pts.flatten(), beta, maxiter=1000, gtol=1e-6)
        x_opt = x_opt.reshape(N, 2)
        # Clamp to disk
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.99
        if np.any(mask):
            x_opt[mask] *= (0.99 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
    
    # Phase 2: Local refinement with higher beta
    betas_phase2 = [1000, 2000, 5000]
    for beta in betas_phase2:
        if time.time() - start_time > DEADLINE - 15:
            break
        x_opt = _optimize_lbfgs(best_pts.flatten(), beta, maxiter=500, gtol=1e-8)
        x_opt = x_opt.reshape(N, 2)
        r = np.sqrt(np.sum(x_opt ** 2, axis=1))
        mask = r > 0.995
        if np.any(mask):
            x_opt[mask] *= (0.995 / r[mask])[:, None]
        s = _score(x_opt)
        if s > best_score:
            best_score = s
            best_pts = x_opt.copy()
    
    # Phase 3: High-precision final polish
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        for beta in [10000, 20000, 50000]:
            if time.time() - start_time > DEADLINE - 2:
                break
            x_opt = _optimize_lbfgs(best_pts.flatten(), beta, maxiter=500, gtol=1e-12)
            x_opt = x_opt.reshape(N, 2)
            r = np.sqrt(np.sum(x_opt ** 2, axis=1))
            mask = r > 0.999
            if np.any(mask):
                x_opt[mask] *= (0.999 / r[mask])[:, None]
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_pts = x_opt.copy()
    
    # Try a few random restarts if time permits
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 20:
        for trial in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            x0 = _random_init(seed=100 + trial).flatten()
            for beta in [200, 1000]:
                if time.time() - start_time > DEADLINE - 5:
                    break
                x_opt = _optimize_lbfgs(x0, beta, maxiter=500, gtol=1e-8)
                x_opt = x_opt.reshape(N, 2)
                r = np.sqrt(np.sum(x_opt ** 2, axis=1))
                mask = r > 0.995
                if np.any(mask):
                    x_opt[mask] *= (0.995 / r[mask])[:, None]
                s = _score(x_opt)
                if s > best_score:
                    best_score = s
                    best_pts = x_opt.copy()
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        best_pts = _hex_lattice_init()
    
    return best_pts
