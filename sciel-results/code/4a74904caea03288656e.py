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

def _soft_min_objective(pts_flat, alpha=50.0):
    """Differentiable soft-min surrogate for min triangle area / hull area."""
    pts = pts_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        
        n = len(pts)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Soft minimum using log-sum-exp
        min_area = np.min(areas)
        if min_area < 1e-10:
            return 1e6
        
        # Normalize for numerical stability
        shifted = areas - min_area
        log_sum_exp = np.log(np.sum(np.exp(-alpha * shifted))) / alpha
        soft_min_area = min_area - log_sum_exp
        
        if soft_min_area <= 0:
            return 1e6
        
        return -(soft_min_area / ha)
    except Exception:
        return 1e6

def _soft_min_objective_grad(pts_flat, alpha=50.0):
    """Numerical gradient of soft-min objective (finite differences)."""
    eps = 1e-6
    grad = np.zeros_like(pts_flat)
    n_params = len(pts_flat)
    
    for i in range(n_params):
        pts_plus = pts_flat.copy()
        pts_minus = pts_flat.copy()
        pts_plus[i] += eps
        pts_minus[i] -= eps
        grad[i] = (_soft_min_objective(pts_plus, alpha) - _soft_min_objective(pts_minus, alpha)) / (2 * eps)
    
    return grad

def _optimize_with_lbfgs(points0, alpha=50.0, max_iter=200):
    """Optimize using L-BFGS-B with soft-min surrogate."""
    pts_flat = points0.flatten()
    n = len(points0)
    bounds = [(0.01, 0.99)] * (2 * n)
    
    try:
        result = minimize(
            _soft_min_objective,
            pts_flat,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if result.x is not None:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points0

def _critical_triangle_polish(points0, max_sweeps=50):
    """Fast polish: only optimize based on the worst (minimum area) triangles."""
    pts = points0.copy()
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
            for step in [0.002, 0.001, 0.0005, 0.0002]:
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

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _good_starting_config(n=13):
    """Create a good starting configuration with hull points on a circle and interior points spread out."""
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
    
    # Strategy 1: Good starting config + L-BFGS-B with soft-min
    start = _good_starting_config(n)
    
    # First pass: L-BFGS-B with moderate alpha
    if time.time() < _DEADLINE - 100:
        opt_pts = _optimize_with_lbfgs(start, alpha=30.0, max_iter=150)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    # Second pass: L-BFGS-B with higher alpha for sharper min
    if best_points is not None and time.time() < _DEADLINE - 80:
        opt_pts2 = _optimize_with_lbfgs(best_points, alpha=80.0, max_iter=100)
        s = _score(opt_pts2)
        if s > best_score:
            best_score = s
            best_points = opt_pts2.copy()
    
    # Critical triangle polish
    if best_points is not None and time.time() < _DEADLINE - 30:
        polished_pts, polished_s = _critical_triangle_polish(best_points, max_sweeps=100)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Try different starting configurations
    if best_score < 0.15 and time.time() < _DEADLINE - 120:
        # Regular 13-gon
        start2 = _regular_polygon(n, radius=0.45, center=(0.5, 0.5))
        opt_pts3 = _optimize_with_lbfgs(start2, alpha=50.0, max_iter=100)
        s = _score(opt_pts3)
        if s > best_score:
            best_score = s
            best_points = opt_pts3.copy()
        
        # Polish
        if best_points is not None and time.time() < _DEADLINE - 30:
            polished_pts2, polished_s2 = _critical_triangle_polish(best_points, max_sweeps=50)
            if polished_s2 > best_score:
                best_score = polished_s2
                best_points = polished_pts2
    
    # Differential evolution as backup if still low score
    if best_score < 0.12 and time.time() < _DEADLINE - 150:
        def de_obj(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            try:
                sc = _score(pts)
                return -sc if sc > 0 else 1.0
            except Exception:
                return 1.0
        
        bounds = [(0.01, 0.99)] * (2 * n)
        
        def time_cb(xk, fun, conv):
            return time.time() > _DEADLINE - 60
        
        try:
            result = differential_evolution(
                de_obj, bounds, seed=42, maxiter=300, popsize=12,
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
        
        # Polish DE result
        if best_points is not None and time.time() < _DEADLINE - 30:
            polished_pts3, polished_s3 = _critical_triangle_polish(best_points, max_sweeps=50)
            if polished_s3 > best_score:
                best_score = polished_s3
                best_points = polished_pts3
    
    # Final polish
    if best_points is not None and time.time() < _DEADLINE - 15:
        final_pts, final_s = _critical_triangle_polish(best_points, max_sweeps=200)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
