import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
import itertools
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 585.0  # Safety margin
_COMBOS = np.array(list(itertools.combinations(range(13), 3)))
_N_COMBOS = len(_COMBOS)

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _ratio(points):
    """Compute min_triangle_area / hull_area. Returns 0 if invalid."""
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma < 1e-12:
        return 0.0
    return ma / ha

def _neg_ratio(points):
    """Negative ratio for minimization."""
    return -_ratio(points)

def _local_refine(points, max_iter=200):
    """Refine using Nelder-Mead."""
    if time.time() > _DEADLINE - 30:
        return points
    
    params = points.flatten()
    try:
        res = minimize(_neg_ratio, params, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
        opt_points = res.x.reshape((13, 2))
        if _ratio(opt_points) > _ratio(points):
            return opt_points
    except Exception:
        pass
    return points

def _optimize_with_de(bounds, max_iter=100, popsize=20, seed=42):
    """Optimize using Differential Evolution."""
    if time.time() > _DEADLINE - 60:
        return None
    
    try:
        res = differential_evolution(_neg_ratio, bounds, maxiter=max_iter, 
                                      popsize=popsize, seed=seed, 
                                      polish=False, tol=1e-6, mutation=(0.5, 1.5),
                                      recombination=0.9)
        return res.x.reshape((13, 2))
    except Exception:
        return None

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _generate_initial_guesses():
    """Generate diverse initial guesses for DE or local refinement."""
    guesses = []
    
    # 1. Regular 13-gon
    guesses.append(_regular_polygon(13, radius=1.0))
    
    # 2. Concentric rings
    for r_in in [0.3, 0.4, 0.5]:
        for n_in in [3, 4, 5, 6]:
            n_out = 13 - n_in
            inner = _regular_polygon(n_in, radius=r_in)
            outer = _regular_polygon(n_out, radius=1.0)
            # Rotate inner slightly
            angle_offset = np.pi / n_in
            inner_rot = np.column_stack([
                r_in * np.cos(np.linspace(angle_offset, 2*np.pi+angle_offset, n_in, endpoint=False)),
                r_in * np.sin(np.linspace(angle_offset, 2*np.pi+angle_offset, n_in, endpoint=False))
            ])
            guesses.append(np.vstack([inner_rot, outer]))
            
    # 3. Random points inside unit circle
    rng = np.random.default_rng(123)
    for _ in range(10):
        angles = rng.uniform(0, 2*np.pi, 13)
        radii = rng.uniform(0.5, 1.0, 13)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        guesses.append(pts)
        
    return guesses

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Bounds for DE: points should be within [-1.5, 1.5] to allow hull to be larger than unit circle if needed
    # But typically optimal is normalized to unit hull. Let's restrict to [-1.1, 1.1] to keep scale reasonable.
    bounds = [(-1.1, 1.1)] * (2 * n)
    
    # Phase 1: Global search with DE
    # We run DE for a fixed number of iterations.
    # To utilize time, we can run multiple DE instances with different seeds if time permits.
    
    start_time = time.time()
    time_budget_for_de = 300  # Spend 300s on DE
    
    # Run DE
    de_points = _optimize_with_de(bounds, maxiter=50, popsize=20, seed=42)
    if de_points is not None:
        de_points = _local_refine(de_points, max_iter=100)
        score = _ratio(de_points)
        if score > best_score:
            best_score = score
            best_points = de_points.copy()
            
    # Phase 2: Local refinement from structured guesses
    # If DE didn't finish or if we have time left, try local refinement from good guesses.
    
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > _DEADLINE - 30:
            break
        
        # Quick local refinement
        refined = _local_refine(guess, max_iter=100)
        score = _ratio(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
            
    # Phase 3: If time remains, try another DE run with different seed
    if time.time() < _DEADLINE - 120:
        de_points2 = _optimize_with_de(bounds, maxiter=30, popsize=20, seed=123)
        if de_points2 is not None:
            de_points2 = _local_refine(de_points2, max_iter=100)
            score = _ratio(de_points2)
            if score > best_score:
                best_score = score
                best_points = de_points2.copy()
                
    # Fallback
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, radius=1.0)
        
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points
