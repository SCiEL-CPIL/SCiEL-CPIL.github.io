"""Combined solver: Voronoi-Lloyd equalization + soft-min polish + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _lloyd_step(c, deadline, steps=200, damping=0.5):
    """Voronoi-based Lloyd's algorithm: move each center toward the centroid
    of its Voronoi cell, clipped to stay inside the cube."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        try:
            tri = Delaunay(c)
        except Exception:
            d = _pair_dist_matrix(c)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            diff = c[:, None, :] - c[None, :, :]
            F = np.sum(diff * inv[:, :, None], axis=1)
            c = np.clip(c + 0.01 * F, -0.49, 0.49)
            continue
        
        simplices = tri.simplices
        
        cell_centroids = np.zeros((N, 3))
        cell_counts = np.zeros(N)
        
        for idx in range(len(simplices)):
            si = simplices[idx]
            v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
            
            A = np.array([
                v1 - v0,
                v2 - v0,
                v3 - v0
            ])
            b = np.array([
                np.dot(v1, v1) - np.dot(v0, v0),
                np.dot(v2, v2) - np.dot(v0, v0),
                np.dot(v3, v3) - np.dot(v0, v0)
            ])
            try:
                cc = np.linalg.solve(2.0 * A, b)
            except np.linalg.LinAlgError:
                cc = np.mean([v0, v1, v2, v3])
            
            for vi in si:
                cell_centroids[vi] += cc
                cell_counts[vi] += 1
        
        valid = cell_counts > 0
        if np.any(valid):
            targets = cell_centroids[valid] / cell_counts[valid, None]
            c[valid] = c[valid] + damping * (targets - c[valid])
        
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _lloyd_optimize(c, deadline):
    """Iterate Lloyd's algorithm with decreasing damping."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    
    for damp in (0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05):
        if time.time() > deadline:
            break
        steps = max(50, int(300 * damp))
        c = _lloyd_step(c, deadline, steps=steps, damping=damp)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()
    
    return best, best_r


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _smooth_polish(c, deadline):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 50},
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


def _repulsion_anneal(c, deadline, steps=3000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
        
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        
        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _local_search(c, deadline, iters=4000):
    """Local search: perturb binding constraints, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
                step = 0.001
                c[i] += direction * step
                c[j] -= direction * step
            else:
                c[i] += rng.uniform(-0.01, 0.01, 3)
                c[j] += rng.uniform(-0.01, 0.01, 3)
        else:
            i, k = min_wall_idx
            direction = np.zeros(3)
            direction[k] = -np.sign(c[i, k])
            step = 0.001
            c[i] += direction * step
        
        c = np.clip(c, -0.499, 0.499)
        
        new_r = _compute_radius(c)
        if new_r > best_r:
            best_r = new_r
            best_c = c.copy()
        else:
            c = best_c.copy()
            break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    best_score = -1.0
    best_P = None
    
    # Initial configurations
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base_grid = grid[:N]
    
    inits = []
    for scale in [0.01, 0.05, 0.1]:
        jitter = np.random.uniform(-scale, scale, size=base_grid.shape)
        P_init = np.clip(base_grid + jitter, -0.5, 0.5)
        inits.append(P_init)
    
    for _ in range(5):
        P_rand = np.random.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(P_rand)

    # Optimization loop
    for i, P_init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Phase 1: Lloyd's algorithm
        P_lloyd, r_lloyd = _lloyd_optimize(P_init, deadline)
        
        # Phase 2: Smooth polish
        P_smooth, r_smooth = _smooth_polish(P_lloyd, deadline)
        
        # Phase 3: Local search
        P_local, r_local = _local_search(P_smooth, deadline)
        
        # Track best
        for P, r in [(P_lloyd, r_lloyd), (P_smooth, r_smooth), (P_local, r_local)]:
            if r > best_score:
                best_score = r
                best_P = P.copy()
    
    if best_P is None:
        return base_grid
        
    return best_P

# EVOLVE-BLOCK-END
