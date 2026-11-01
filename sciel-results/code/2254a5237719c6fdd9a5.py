"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the common radius for a given set of centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    max_coord = np.max(np.abs(pts), axis=1)
    if np.any(max_coord > CUBE_HALF + 1e-12):
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist <= 1e-12:
        return 0.0
    r_pair = min_dist / 2.0
    r_wall = CUBE_HALF - np.max(max_coord)
    return min(r_pair, r_wall)


def _softmin_radius_and_grad(pts, beta):
    """Compute soft-min radius and its gradient w.r.t. pts.
    
    Returns (r_soft, grad_r_soft) where r_soft approximates the true min radius.
    """
    pts = np.asarray(pts).reshape(N, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall margins: distance from each point to each face
    wall_margins = CUBE_HALF - np.abs(pts)  # shape (N, 3)
    
    # Collect all radius candidates
    tri_idx = np.triu_indices(N, k=1)
    pair_radii = dists[tri_idx] / 2.0
    wall_radii = wall_margins.ravel()
    
    all_radii = np.concatenate([pair_radii, wall_radii])
    
    # Numerically stable soft min
    min_val = np.min(all_radii)
    shifted = all_radii - min_val
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    r_soft = min_val - log_sum / beta
    
    # Weights for gradient
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of r_soft w.r.t. pts
    grad = np.zeros_like(pts)
    
    # Pair contributions
    n_pairs = len(pair_radii)
    for k, (i, j) in enumerate(zip(*tri_idx)):
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = weights[k]
        # d(pair_radius)/d(pts[i]) = (pts[i] - pts[j]) / (2 * d)
        vec = (pts[i] - pts[j]) / (2.0 * d)
        grad[i] += w * vec
        grad[j] -= w * vec
    
    # Wall contributions
    wall_offset = n_pairs
    for i in range(N):
        for a in range(3):
            k = wall_offset + i * 3 + a
            w = weights[k]
            # d(CUBE_HALF - |pts[i,a]|)/d(pts[i,a]) = -sign(pts[i,a])
            sgn = 1.0 if pts[i, a] >= 0 else -1.0
            grad[i, a] += w * (-sgn)
    
    return r_soft, grad


def _epigraph_objective(pts, t, beta):
    """Epigraph formulation: minimize -t + penalty for t > r_soft.
    
    We want to maximize t subject to t <= r_soft (approx).
    Objective: -t + (1/beta) * log(sum(exp(beta*(t - r_i)))) for all constraints r_i.
    This is equivalent to maximizing the soft min.
    """
    pts = np.asarray(pts).reshape(N, 3)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    wall_margins = CUBE_HALF - np.abs(pts)
    
    tri_idx = np.triu_indices(N, k=1)
    pair_radii = dists[tri_idx] / 2.0
    wall_radii = wall_margins.ravel()
    
    all_radii = np.concatenate([pair_radii, wall_radii])
    
    # Soft min of all_radii
    min_val = np.min(all_radii)
    shifted = all_radii - min_val
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    r_soft = min_val - log_sum / beta
    
    # We want to maximize r_soft, so minimize -r_soft
    # But also ensure t tracks r_soft
    # Simple approach: just minimize -r_soft, ignore t
    obj = -r_soft
    
    # Gradient w.r.t. pts
    weights = exp_vals / np.sum(exp_vals)
    grad = np.zeros_like(pts)
    
    n_pairs = len(pair_radii)
    for k, (i, j) in enumerate(zip(*tri_idx)):
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = weights[k]
        vec = (pts[i] - pts[j]) / (2.0 * d)
        grad[i] += w * vec
        grad[j] -= w * vec
    
    wall_offset = n_pairs
    for i in range(N):
        for a in range(3):
            k = wall_offset + i * 3 + a
            w = weights[k]
            sgn = 1.0 if pts[i, a] >= 0 else -1.0
            grad[i, a] += w * (-sgn)
    
    # Objective is -r_soft, so gradient is -grad
    obj_grad = -grad
    
    return obj, obj_grad


def _repulsion_anneal(pts, start_time, deadline, n_steps=100):
    """Repulsion-based annealing to spread points apart."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for step in range(n_steps):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        # Compute repulsive forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion: force = direction / dist^2 (Coulomb-like)
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-12:
                    continue
                # Only consider close pairs for efficiency
                if d < 0.5:
                    forces[i] += (pts[i] - pts[j]) / (d * d * d)
        
        # Wall repulsion: push points away from walls
        for i in range(N):
            for a in range(3):
                margin = CUBE_HALF - abs(pts[i, a])
                if margin < 0.1:
                    sgn = 1.0 if pts[i, a] >= 0 else -1.0
                    forces[i, a] += -sgn * (0.1 - margin) ** 2
        
        # Step size annealing
        t = step / max(n_steps - 1, 1)
        step_size = 0.05 * (1.0 - t) ** 1.5
        
        pts_new = pts + step_size * forces
        pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    return best_pts, best_r


def _local_search(pts, start_time, deadline, max_iters=50):
    """Local search with small random perturbations."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        # Random perturbation
        scale = 0.02 * (1.0 - it / max_iters)
        perturbation = rng.normal(0, scale, size=pts.shape)
        pts_new = pts + perturbation
        pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
    
    return best_pts, best_r


def _polish(pts, start_time, deadline):
    """Final high-precision polish with increasing beta."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        def obj_func(p):
            p = p.reshape(N, 3)
            f, g = _epigraph_objective(p, 0.0, beta)
            return f, g.ravel()
        
        try:
            res = minimize(obj_func, best_pts.ravel(), jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-12})
            pts_new = res.x.reshape(N, 3)
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            r_new = _compute_radius(pts_new)
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_new.copy()
        except Exception:
            pass
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial placement: FCC-like lattice
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    best_pts = None
    best_r = -1.0
    
    # Configuration 1: Perturbed grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts1 = grid[:N].copy()
    pts1 += rng.normal(0, 0.01, size=pts1.shape)
    pts1 = np.clip(pts1, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
    
    # Configuration 2: Random
    pts2 = rng.uniform(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, size=(N, 3))
    
    # Configuration 3: FCC-like
    a = 0.35
    fcc_sites = []
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                fcc_sites.append((ix * a, iy * a, iz * a))
                fcc_sites.append(((ix + 0.5) * a, (iy + 0.5) * a, iz * a))
                fcc_sites.append((ix * a, (iy + 0.5) * a, (iz + 0.5) * a))
                fcc_sites.append(((ix + 0.5) * a, iy * a, (iz + 0.5) * a))
    fcc_sites = np.array(fcc_sites)
    # Filter to those inside cube
    mask = np.all(np.abs(fcc_sites) < CUBE_HALF - 0.02, axis=1)
    fcc_sites = fcc_sites[mask]
    if len(fcc_sites) >= N:
        pts3 = fcc_sites[:N].copy()
        pts3 += rng.normal(0, 0.005, size=pts3.shape)
        pts3 = np.clip(pts3, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
    else:
        pts3 = pts1.copy()
    
    for pts_init in [pts1, pts2, pts3]:
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Repulsion annealing
        pts_r, r_r = _repulsion_anneal(pts_init.copy(), start_time, deadline, n_steps=50)
        
        # Phase 2: Gradient-based optimization
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        pts_opt = pts_r.copy()
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 20:
                break
            
            def obj_func(p, b=beta):
                p = p.reshape(N, 3)
                f, g = _epigraph_objective(p, 0.0, b)
                return f, g.ravel()
            
            try:
                res = minimize(obj_func, pts_opt.ravel(), jac=True, bounds=bounds,
                              method='L-BFGS-B', options={'maxiter': 30, 'gtol': 1e-8})
                pts_new = res.x.reshape(N, 3)
                pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
                r_new = _compute_radius(pts_new)
                if r_new > r_r:
                    r_r = r_new
                    pts_opt = pts_new
            except Exception:
                pass
        
        # Phase 3: Local search
        pts_ls, r_ls = _local_search(pts_opt, start_time, deadline, max_iters=20)
        
        # Phase 4: Polish
        pts_pol, r_pol = _polish(pts_ls, start_time, deadline)
        
        if r_pol > best_r:
            best_r = r_pol
            best_pts = pts_pol.copy()
    
    if best_pts is None:
        best_pts = pts1
    
    return best_pts
