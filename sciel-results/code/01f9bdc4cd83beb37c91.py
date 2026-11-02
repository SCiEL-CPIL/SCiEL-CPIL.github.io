"""
Combined solver: FCC Farthest-Point Seed + Vectorized Repulsion + Annealed Soft-Min Polish + Exact Nelder-Mead.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5, 0.5]^3.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE_MARGIN = 12.0  # Safety margin before 150s kill

def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5, 0.5]^3."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return -1.0
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -1.0
    # Compute pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Compute distance to walls
    # c is in [-0.5, 0.5], so distance to wall is 0.5 - |c|
    min_wall = np.min(0.5 - np.abs(c))
    
    return min(min_pair, min_wall)

def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    return d

def _softmin_radius(c, t):
    """
    Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). Returns a smooth lower-bound-ish value
    that increases as the packing tightens. t = temperature (small = sharp).
    """
    d = _pair_dist_matrix(c)
    # Use only upper triangle to avoid double-counting pairs
    iu = np.triu_indices(N, k=1)
    pair = d[iu] / 2.0
    
    # Wall constraints: 0.5 - |c_ik|
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    vals = np.concatenate([pair, wall_flat])
    vals = vals[np.isfinite(vals)]
    
    if len(vals) == 0:
        return 0.0
        
    m = np.min(vals)
    # Numerical stability: shift by min
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)

def _smooth_obj(x, t):
    """
    Objective to MINIMIZE: negative smooth radius plus a small wall-push penalty.
    """
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    # Penalty for going out of bounds (should be rare if bounds are enforced in optimizer)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen

def _neg_score(centers):
    """Negative exact score for minimisation."""
    return -_compute_radius(centers.reshape(N, 3))

def _time_left(deadline):
    return deadline - time.time()

def _repulsion_anneal(c, deadline, steps=None):
    """
    Vectorized repulsion + wall-attraction gradient descent.
    This is a fast, coarse-grained optimizer to improve the initial configuration.
    """
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    if steps is None:
        steps = 3000
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        d = _pair_dist_matrix(c)
        # Repulsion force ~ 1/d^2
        # Avoid division by zero
        inv_d2 = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F_rep = np.sum(diff * inv_d2[:, :, None], axis=1)
        
        # Wall force: push towards center if close to wall
        # F_wall ~ (0.5 - |c|) * sign(c)
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            dist_to_wall = 0.5 - np.abs(c[:, k])
            # Stronger push if very close
            wall_force[:, k] = sgn * (dist_to_wall ** 2)
        
        frac = it / max(steps - 1, 1)
        # Annealing: decrease repulsion, increase wall attraction
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        
        c = c + lr * (rep_scale * F_rep + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
        
    return c

def _smooth_polish(c, deadline):
    """
    Polish a configuration with the smooth softmin max-min via L-BFGS-B,
    annealing the temperature down for a sharp min.
    """
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    # Temperature schedule: from smooth (global) to sharp (local)
    temps = (0.05, 0.02, 0.008, 0.003, 0.001, 0.0004, 0.0001)
    
    for t in temps:
        if _time_left(deadline) < 10.0:
            break
        
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 60},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
            
    return best, best_r

def _exact_polish(c, deadline):
    """
    Nelder-Mead on the exact min-radius objective, very tight.
    """
    c = np.clip(c.copy(), -0.49, 0.49)
    best = c
    best_r = _compute_radius(c)

    # First pass
    try:
        res = minimize(
            _neg_score, c.flatten(), method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True},
        )
        c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c
    except Exception:
        pass

    # Second pass if time allows
    if _time_left(deadline) > 15.0:
        try:
            res = minimize(
                _neg_score, best.flatten(), method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c
        except Exception:
            pass

    return best, best_r

def _seed_fcc():
    """
    FCC lattice supercell, then farthest-point sample to N points.
    This provides a highly uniform and symmetric starting configuration.
    """
    pts = []
    a = 1.0
    # 3x3x3 supercell of FCC lattice
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
                
    pts = np.array(pts)
    # Center and scale to fit within [-0.45, 0.45]
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
        
    # Farthest point sampling
    # Start with the point closest to the center (0,0,0)
    dists_to_center = np.sum(pts**2, axis=1)
    sel = [int(np.argmin(dists_to_center))]
    rem = [i for i in range(len(pts)) if i != sel[0]]
    
    while len(sel) < N and rem:
        if len(sel) > 1:
            # Compute min distance from each candidate in 'rem' to the selected points 'sel'
            # To be efficient, we can just compute dists between rem and sel
            dm = np.min(cdist(pts[rem], pts[sel]), axis=1)
        else:
            # If only 1 selected, just use distance to that point
            dm = cdist(pts[rem], pts[sel[0]])[:, 0]
            
        # Select the candidate with the maximum minimum distance
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
        
    return pts[sel]

def _seed_grid():
    """Fallback grid seed."""
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_centers = None
    best_radius = -np.inf
    
    def _evaluate(centers):
        nonlocal best_centers, best_radius
        if centers is None or not np.all(np.isfinite(centers)):
            return
        if np.any(np.abs(centers) > 0.5):
            return
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()

    # 1. Generate Seed
    current = _seed_fcc()
    _evaluate(current)
    
    # 2. Coarse Optimization: Repulsion Annealing
    if _time_left(deadline) > 40:
        current = _repulsion_anneal(current, deadline - 40)
        _evaluate(current)
        
    # 3. Medium Optimization: Annealed Smooth Polish
    if _time_left(deadline) > 30:
        current, _ = _smooth_polish(current, deadline - 30)
        _evaluate(current)
        
    # 4. Fine Optimization: Exact Nelder-Mead
    if _time_left(deadline) > 10:
        current, _ = _exact_polish(current, deadline)
        _evaluate(current)
        
    # 5. Optional: Second run with a different seed if time allows
    if _time_left(deadline) > 30:
        # Use a slightly perturbed FCC seed or a different seed
        current2 = _seed_fcc()
        # Perturb to break symmetry
        rng = np.random.default_rng(42)
        current2 += rng.normal(0, 0.01, size=(N, 3))
        current2 = np.clip(current2, -0.49, 0.49)
        _evaluate(current2)
        
        if _time_left(deadline) > 25:
            current2 = _repulsion_anneal(current2, deadline - 25)
            _evaluate(current2)
            
        if _time_left(deadline) > 15:
            current2, _ = _smooth_polish(current2, deadline - 15)
            _evaluate(current2)
            
        if _time_left(deadline) > 5:
            current2, _ = _exact_polish(current2, deadline)
            _evaluate(current2)

    if best_centers is None:
        return _seed_grid()
        
    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    return best_centers
