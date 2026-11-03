import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
DEADLINE = START_TIME + 135.0  # Leave 15s margin before 150s kill

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _spherical_circumradius_fast(v1, v2, v3):
    """
    Compute the spherical circumradius of the triangle formed by v1, v2, v3 on the unit sphere.
    Returns the angle in radians. Uses the cross-product method for speed.
    """
    d1 = v1 - v2
    d2 = v2 - v3
    c = np.cross(d1, d2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        return np.pi / 2
    c = c / norm_c
    dot = np.dot(c, v1)
    if dot < 0:
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facets using fast spherical circumradius."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex_idx in hull.simplices:
            if len(simplex_idx) == 3:
                simplex = pts[simplex_idx]
                cr = _spherical_circumradius_fast(simplex[0], simplex[1], simplex[2])
                if cr > max_cr:
                    max_cr = cr
        return np.degrees(max_cr)
    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        samples = rng.randn(2000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dots = np.clip(np.dot(samples, pts.T), -1, 1)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        return np.degrees(np.max(min_angles))

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_vertices():
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    return _normalize(ico)

def _greedy_farthest_point(all_pts, n_select, seed=0):
    """Select n_select points from all_pts using greedy farthest-point sampling."""
    rng = np.random.RandomState(seed)
    m = len(all_pts)
    if m <= n_select:
        return all_pts.copy()
    
    # Start with a random point
    selected = [rng.randint(m)]
    
    for _ in range(n_select - 1):
        sel_pts = all_pts[selected]
        # Compute angular distance from each candidate to nearest selected
        dots = np.clip(np.dot(all_pts, sel_pts.T), -1, 1)
        angles = np.arccos(dots)
        min_dists = np.min(angles, axis=1)
        # Exclude already selected
        for s in selected:
            min_dists[s] = -1
        selected.append(int(np.argmax(min_dists)))
    
    return all_pts[selected]

def _icosahedron_based_init(n, seed):
    """Generate n points starting from icosahedron vertices + random, then greedy select."""
    ico = _icosahedron_vertices()
    rng = np.random.RandomState(seed)
    extra = rng.randn(n, 3)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    all_pts = np.vstack([ico, extra])
    return _greedy_farthest_point(all_pts, n, seed)

def _nelder_mead_refine(pts, time_budget):
    """Refine points using Nelder-Mead on the covering radius."""
    pts = _normalize(pts).copy()
    best = pts.copy()
    best_cr = _covering_radius(best)
    start_time = time.time()
    
    def objective_flat(x):
        p = _normalize(x.reshape(-1, 3))
        return _covering_radius(p)
    
    # Multiple restarts of Nelder-Mead
    for restart in range(10):
        if time.time() - start_time > time_budget - 5:
            break
        x0 = best.flatten()
        try:
            res = minimize(
                objective_flat, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                         'initial_simplex': None, 'adaptive': True}
            )
            if res.fun < best_cr:
                best_cr = res.fun
                best = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
        if time.time() - start_time > time_budget - 5:
            break
    
    return best, best_cr

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Generate diverse initializations
    initializations = []
    initializations.append(('fib', _fibonacci_points(N)))
    for seed in range(5):
        initializations.append((f'rand_{seed}', _random_points(N, seed)))
    for seed in range(5):
        initializations.append((f'ico_{seed}', _icosahedron_based_init(N, seed)))
    
    best_pts = None
    best_cr = np.inf
    
    # Evaluate all initializations
    for name, pts in initializations:
        if time.time() > deadline - 20:
            break
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Sort initializations by CR to prioritize good ones for refinement
    init_scores = []
    for name, pts in initializations:
        if time.time() > deadline - 15:
            break
        cr = _covering_radius(pts)
        init_scores.append((cr, name, pts))
    init_scores.sort(key=lambda x: x[0])
    
    # Refine top initializations with Nelder-Mead
    if best_pts is not None:
        for cr, name, pts in init_scores:
            if time.time() > deadline - 10:
                break
            if cr > best_cr + 2.0:
                break  # Skip if too far from best
            time_budget = min(30, deadline - time.time() - 5)
            if time_budget < 10:
                break
            refined_pts, refined_cr = _nelder_mead_refine(pts, time_budget)
            if refined_cr < best_cr:
                best_cr = refined_cr
                best_pts = refined_pts
    
    # Final polish on the best
    if best_pts is not None and time.time() < deadline - 10:
        time_budget = min(20, deadline - time.time() - 5)
        if time_budget > 5:
            refined_pts, refined_cr = _nelder_mead_refine(best_pts, time_budget)
            if refined_cr < best_cr:
                best_cr = refined_cr
                best_pts = refined_pts
    
    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    best_pts = _normalize(best_pts)
    
    # Validate: check no coincident points
    rng = np.random.RandomState(123)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] = best_pts[j] * 0.999 + 0.001 * rng.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
