"""
Optimization approach: Multi-start local search followed by global differential evolution.
Focuses on robustness and escaping local minima common in non-convex geometric optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    # Iterate over pairs (i, j) and vectorize over k > j
    # This avoids computing all N choose 3 triples explicitly in a slow loop
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectors from i to j and i to k
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            # Vectors from i to k for all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product magnitude / 2 gives area
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or degenerate, area is 0
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(flat):
    """
    Objective function for minimization.
    We want to maximize the score, so we minimize the inverse.
    If the configuration is degenerate, return a large penalty.
    """
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
        
    # Minimize hull_area / min_area is equivalent to maximizing min_area / hull_area
    return hull_area / min_area

def _generate_initial_configs():
    """
    Generate a diverse set of initial configurations.
    """
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two concentric rings (7 + 7)
    # Ring 1: radius 0.5, Ring 2: radius 1.0
    for r_inner in [0.3, 0.5, 0.7]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7 # Staggered
        p1 = r_inner * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([np.cos(a2), np.sin(a2)])
        configs.append(np.vstack([p1, p2]))
        
    # 3. Three rings (5 + 4 + 5) or other splits
    # 5 on outer, 4 on middle, 5 on inner? Or 6+4+4?
    # Let's try 6+4+4
    a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = 0.6 * np.column_stack([np.cos(a2), np.sin(a2)])
    p3 = 0.3 * np.column_stack([np.cos(a3), np.sin(a3)])
    configs.append(np.vstack([p1, p2, p3]))
    
    # 4. Random configurations
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Uniform in disk
        r = rng.uniform(0, 1, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        configs.append(pts)
        
    # 5. Jittered regular polygon
    base = configs[0]
    for std in [0.05, 0.1, 0.2]:
        jitter = rng.normal(0, std, size=(N, 2))
        configs.append(base + jitter)
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective(pts.flatten())

    # Phase 1: Multi-start Nelder-Mead
    inits = _generate_initial_configs()
    for i, pts0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, pts0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success or res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 2: Differential Evolution (Global Search)
    # DE is robust but slow. We use a small population and limited iterations.
    if time.time() < deadline - 20:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            # Use a seed for reproducibility
            de_res = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=50, tol=1e-8,
                popsize=10, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol' # Sobol sequence for better initial distribution
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 3: Final Polish
    # Refine the best candidate found so far
    if best_pts is not None and time.time() < deadline - 5:
        try:
            # Try a few different local optimizers on the best point
            # 1. Nelder-Mead with tighter tolerance
            res = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
                
            # 2. Powell's Method
            if time.time() < deadline - 2:
                res_powell = minimize(_objective, best_pts.flatten(), method='Powell',
                                      options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14})
                if res_powell.fun < best_obj:
                    update_best(res_powell.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize to unit area
    # The score is invariant to scaling, but the task says "normalized to unit area".
    # We should return points such that their convex hull has area 1.
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass

    return best_pts
