# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _vertex_score_from_dists(dists):
    """Compute vertex score given sorted distances to other 19 vertices."""
    if np.any(dists < 0.05):
        return 0.0
    
    diffs = np.diff(dists)
    k = np.argmin(diffs)
    D = (dists[k] + dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    d_sel = dists[idxs]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    """Compute the minimum vertex score for a normalized polygon."""
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    radii = np.linalg.norm(poly, axis=1)
    mean_r = np.mean(radii)
    if mean_r <= 0.01:
        return 0.0
    poly = poly / mean_r
    
    min_score = 1.0
    for i in range(n):
        v = poly[i]
        diffs = poly - v
        dists = np.linalg.norm(diffs, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists = dists[mask]
        dists = np.sort(dists)
        
        s = _vertex_score_from_dists(dists)
        if s < min_score:
            min_score = s
            
    return min_score

def _check_convex(polygon):
    """Check if polygon is strictly convex with CCW orientation."""
    n = len(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return np.all(cross > 0)

def _build_bipartite(r1, r2, offset):
    """
    Build a 20-gon with bipartite radial structure.
    Vertices at even indices have radius r1, odd indices have radius r2.
    Angular positions are on a regular 10-gon grid with offset.
    """
    n = 20
    base_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    angles = np.concatenate([base_angles + offset, base_angles + np.pi/10 + offset])
    
    # Reorder to get cyclic order: 0, 1, 2, ..., 19
    # Even indices (0,2,4,...) get r1, odd indices (1,3,5,...) get r2
    # In our construction:
    # indices 0-9: angles from first set (base + offset)
    # indices 10-19: angles from second set (base + pi/10 + offset)
    # But we need to interleave them properly.
    
    # Actually, let's think about this differently.
    # We want 20 vertices in cyclic order.
    # Vertex i has angle = i * pi/10 + offset (for i=0,...,19)
    # Even i: radius r1, odd i: radius r2
    
    angles = np.arange(20) * (np.pi / 10) + offset
    radii = np.where(np.arange(20) % 2 == 0, r1, r2)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective_bipartite(params):
    """Objective function for bipartite parameterization."""
    r1, r2, offset = params
    if r1 <= 0 or r2 <= 0:
        return 1000.0
    
    polygon = _build_bipartite(r1, r2, offset)
    
    if not _check_convex(polygon):
        return 1000.0
    
    score = _min_vertex_score(polygon)
    return -score

def _objective_general(theta):
    """Objective function for general angle parameterization."""
    x = np.cos(theta)
    y = np.sin(theta)
    poly = np.stack([x, y], axis=1)
    
    if not _check_convex(poly):
        return 1000.0
    
    score = _min_vertex_score(poly)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -1.0
    best_polygon = None
    
    # Strategy 1: Bipartite radial optimization
    # Initial guess: r1=1, r2=1 (regular polygon), offset=0
    init_bipartite = np.array([1.0, 1.0, 0.0])
    
    try:
        res1 = opt.minimize(_objective_bipartite, init_bipartite, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res1.fun < -best_score:
            best_score = -res1.fun
            best_polygon = _build_bipartite(*res1.x)
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_polygon is not None:
            return best_polygon
        return _build_bipartite(1.0, 1.0, 0.0)
    
    # Try different initial conditions for bipartite
    for r1_init in [0.8, 0.9, 1.1, 1.2]:
        for r2_init in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time.time() > deadline:
                break
            init = np.array([r1_init, r2_init, 0.0])
            try:
                res = opt.minimize(_objective_bipartite, init, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_polygon = _build_bipartite(*res.x)
            except Exception:
                pass
    
    if time.time() > deadline:
        if best_polygon is not None:
            return best_polygon
        return _build_bipartite(1.0, 1.0, 0.0)
    
    # Strategy 2: General 20-angle optimization from best bipartite result
    if best_polygon is not None:
        # Extract angles from best polygon
        theta_init = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        # Ensure angles are in [0, 2*pi)
        theta_init = np.mod(theta_init, 2 * np.pi)
        # Sort to ensure cyclic order
        sort_idx = np.argsort(theta_init)
        theta_init = theta_init[sort_idx]
        
        try:
            res2 = opt.minimize(_objective_general, theta_init, method='Powell',
                               options={'maxiter': 3000, 'maxfev': 20000})
            if res2.fun < -best_score:
                best_score = -res2.fun
                x = np.cos(res2.x)
                y = np.sin(res2.x)
                best_polygon = np.stack([x, y], axis=1)
        except Exception:
            pass
    
    if time.time() > deadline:
        if best_polygon is not None:
            return best_polygon
        return _build_bipartite(1.0, 1.0, 0.0)
    
    # Strategy 3: Refine with Nelder-Mead on general parameters
    if best_polygon is not None:
        theta_init = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        theta_init = np.mod(theta_init, 2 * np.pi)
        sort_idx = np.argsort(theta_init)
        theta_init = theta_init[sort_idx]
        
        try:
            res3 = opt.minimize(_objective_general, theta_init, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res3.fun < -best_score:
                best_score = -res3.fun
                x = np.cos(res3.x)
                y = np.sin(res3.x)
                best_polygon = np.stack([x, y], axis=1)
        except Exception:
            pass
    
    if best_polygon is None:
        # Fallback: regular polygon
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_polygon = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    return best_polygon

# EVOLVE-BLOCK-END
