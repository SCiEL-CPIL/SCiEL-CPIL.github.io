import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Return the minimum triangle area, or 0 if degenerate."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        areas = _triangle_areas(points)
        return float(np.min(areas))
    except Exception:
        return 0.0

def _nelder_mead_optimize(points, deadline=None, max_iter=5000):
    """Optimize using Nelder-Mead on flattened point coordinates."""
    from scipy.optimize import minimize
    
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining < 10:
            return points, _score(points)
    
    def objective(flat_pts):
        pts = flat_pts.reshape(13, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if deadline is not None and time.time() > deadline:
            return points, _score(points)
        
        opt_pts = result.x.reshape(13, 2)
        opt_score = _score(opt_pts)
        
        if opt_score > _score(points):
            return opt_pts, opt_score
        else:
            return points, _score(points)
    except Exception:
        return points, _score(points)

def _generate_initial_points(method='polygon', seed=42):
    """Generate diverse initial point sets with better spread."""
    n = 13
    rng = np.random.default_rng(seed)
    
    if method == 'polygon':
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    elif method == 'grid':
        # 4x4 grid minus one corner, slightly perturbed
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        pts = np.array([[x, y] for x in xs for y in ys])
        pts = pts[:13]
        pts += rng.normal(0, 0.05, size=(13, 2))
        return pts
    elif method == 'random':
        pts = rng.uniform(-1, 1, size=(13, 2))
        return pts
    elif method == 'mixed':
        # Mix of polygon and random
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts += rng.normal(0, 0.1, size=(13, 2))
        return pts
    elif method == 'circle_perturbed':
        # Circle with varying radius perturbation
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = 1 + rng.normal(0, 0.15, size=n)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        return pts
    elif method == 'two_rings':
        # Points on two concentric circles
        n_outer = 7
        n_inner = 6
        outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
        outer_pts = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
        inner_pts = 0.5 * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
        pts = np.vstack([outer_pts, inner_pts])
        pts += rng.normal(0, 0.05, size=(13, 2))
        return pts
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Try multiple initializations with Nelder-Mead optimization
    best_points = None
    best_score = 0.0
    
    methods = ['polygon', 'grid', 'mixed', 'random', 'circle_perturbed', 'two_rings']
    seeds = [42, 123, 456, 789, 101, 202, 303, 777, 999, 111]
    
    for method in methods:
        for seed in seeds:
            if time.time() > deadline - 30:
                break
            
            init_pts = _generate_initial_points(method, seed)
            init_score = _score(init_pts)
            
            # Use Nelder-Mead for optimization
            opt_pts, opt_score = _nelder_mead_optimize(init_pts, deadline=deadline, max_iter=3000)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
            
            # Also track if initial was better (shouldn't happen but just in case)
            if init_score > best_score:
                best_score = init_score
                best_points = init_pts.copy()
        
        if time.time() > deadline - 30:
            break
    
    # If we have time, do a final refinement on the best point
    if best_points is not None and time.time() < deadline - 15:
        refined_pts, refined_score = _nelder_mead_optimize(best_points, deadline=deadline, max_iter=2000)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None or _score(best_points) <= 0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
