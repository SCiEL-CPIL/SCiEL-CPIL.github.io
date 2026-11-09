# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
import itertools

_DEADLINE = time.time() + 615.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _negative_ratio(points_flat):
    """Objective for optimizers: minimize negative score."""
    points = points_flat.reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        return 1e6
    try:
        s = _score(points)
        if s <= 0:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _delaunay_centroid(points):
    """Calculate centroids of Delaunay triangles."""
    try:
        tri = Delaunay(points)
        centroids = np.zeros((len(tri.simplices), 2))
        for i, simplex in enumerate(tri.simplices):
            centroids[i] = np.mean(points[simplex], axis=0)
        return centroids
    except Exception:
        return None

def _lloyd_iteration(points, n_iters=10):
    """
    Perform Lloyd's algorithm style refinement.
    Move each point to the centroid of its Voronoi cell (approximated by Delaunay neighbors).
    This tends to equalize triangle areas.
    """
    pts = points.copy()
    for _ in range(n_iters):
        if time.time() > _DEADLINE - 30:
            break
        try:
            tri = Delaunay(pts)
            # For each point, find the centroid of the triangles it is part of
            # This is a heuristic approximation of the Voronoi centroid
            new_pts = pts.copy()
            for i in range(len(pts)):
                # Find all triangles containing point i
                mask = (tri.simplices == i).any(axis=1)
                if np.any(mask):
                    relevant_tris = tri.simplices[mask]
                    # Get the centroids of these triangles
                    tri_centroids = np.mean(pts[relevant_tris], axis=1)
                    # Move point i towards the centroid of its surrounding triangle centroids
                    # Damping factor to prevent oscillation
                    target = np.mean(tri_centroids, axis=0)
                    new_pts[i] = 0.5 * pts[i] + 0.5 * target
            pts = new_pts
        except Exception:
            break
    return pts

def _nelder_mead_optimize(points, max_iter=1000):
    """
    Use Nelder-Mead to optimize the negative ratio.
    Nelder-Mead is robust to non-smooth functions like min-area.
    """
    if time.time() > _DEADLINE - 60:
        return points
    
    pts_flat = points.flatten()
    
    try:
        result = minimize(
            _negative_ratio,
            pts_flat,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if result.x is not None:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points

def _coordinate_descent_polish(points, max_sweeps=50):
    """
    Fine-tune using coordinate descent on the critical triangles.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
            break
        
        # Find the minimum area triangle
        areas = _triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area < 1e-10:
            break
        
        # Get the three points in the critical triangle
        idx = np.array(list(itertools.combinations(range(n), 3)))
        crit_indices = idx[min_idx]
        
        improved = False
        # Try small moves for each of the 3 critical points
        for pi in crit_indices:
            x0, y0 = pts[pi]
            best_x, best_y = x0, y0
            best_local = best_score
            
            # Try several step sizes
            for step in [0.005, 0.002, 0.001, 0.0005]:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step*0.707, step*0.707), (-step*0.707, step*0.707),
                               (step*0.707, -step*0.707), (-step*0.707, -step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[pi] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_x, best_y = nx, ny
                        improved = True
            
            pts[pi] = [best_x, best_y]
        
        cur = _score(pts)
        if cur > best_score + 1e-12:
            best_score = cur
            best_pts = pts.copy()
        
        if not improved:
            break
    
    return best_pts, best_score

def _triangle_areas(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return areas

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _good_starting_config(n=13):
    """
    Create a good starting configuration.
    Mix of hull points and interior points.
    """
    # 8 hull points on a circle
    n_hull = 8
    hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    hull_pts = np.column_stack([0.5 + 0.45 * np.cos(hull_angles),
                                 0.5 + 0.45 * np.sin(hull_angles)])
    
    # 5 interior points in a smaller pentagon
    n_int = n - n_hull
    int_angles = np.linspace(0, 2 * np.pi, n_int, endpoint=False) + np.pi / n_int
    int_pts = np.column_stack([0.5 + 0.15 * np.cos(int_angles),
                                0.5 + 0.15 * np.sin(int_angles)])
    
    return np.vstack([hull_pts, int_pts])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Differential Evolution for global search
    if time.time() < _DEADLINE - 300:
        bounds = [(0.01, 0.99)] * (2 * n)
        
        def de_obj(pts_flat):
            return _negative_ratio(pts_flat)
        
        def time_cb(xk, fun, conv):
            return time.time() > _DEADLINE - 100
        
        try:
            result = differential_evolution(
                de_obj, bounds, seed=42, maxiter=100, popsize=15,
                tol=1e-4, callback=time_cb, polish=False
            )
            if result.x is not None:
                de_pts = result.x.reshape(-1, 2)
                s = _score(de_pts)
                if s > best_score:
                    best_score = s
                    best_points = de_pts.copy()
        except Exception:
            pass
    
    # Strategy 2: If DE failed or score is low, try good starting config + Nelder-Mead
    if best_score < 0.1 or best_points is None:
        if time.time() < _DEADLINE - 200:
            start = _good_starting_config(n)
            nm_pts = _nelder_mead_optimize(start, max_iter=500)
            s = _score(nm_pts)
            if s > best_score:
                best_score = s
                best_points = nm_pts.copy()
    
    # Strategy 3: Lloyd's algorithm refinement
    if best_points is not None and time.time() < _DEADLINE - 100:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=20)
        s = _score(lloyd_pts)
        if s > best_score:
            best_score = s
            best_points = lloyd_pts.copy()
            
        # Refine Lloyd's result with Nelder-Mead
        if time.time() < _DEADLINE - 60:
            nm_lloyd_pts = _nelder_mead_optimize(lloyd_pts, max_iter=300)
            s = _score(nm_lloyd_pts)
            if s > best_score:
                best_score = s
                best_points = nm_lloyd_pts.copy()

    # Strategy 4: Final Polish with Coordinate Descent
    if best_points is not None and time.time() < _DEADLINE - 30:
        polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=100)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
        
    return best_points

# EVOLVE-BLOCK-END
