# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 625.0

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

def _thomson_objective(pts_flat, alpha=20.0, beta=5.0):
    """
    Thomson-like energy: minimize max pairwise 'inverse distance' energy.
    This encourages points to spread out, which correlates with maximizing min triangle area.
    alpha controls sharpness of max approximation, beta controls hull regularization.
    """
    n = len(pts_flat) // 2
    pts = pts_flat.reshape(-1, 2)
    
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Inverse distances (avoid division by zero)
    inv_dist = 1.0 / np.sqrt(dist_sq + 1e-15)
    
    # Soft max of inverse distances using log-sum-exp
    max_inv = np.max(inv_dist)
    log_sum_exp = np.log(np.sum(np.exp(alpha * (inv_dist - max_inv)))) / alpha + max_inv
    
    # Hull regularization: penalize points that are too close to boundary or outside
    # Use a soft constraint: penalize if point is outside [0.05, 0.95] box
    x = pts[:, 0]
    y = pts[:, 1]
    hull_penalty = np.sum(np.maximum(0, x - 0.95)**2 + np.maximum(0.05 - x)**2 +
                          np.maximum(0, y - 0.95)**2 + np.maximum(0.05 - y)**2)
    
    # Also penalize very close points (collision)
    min_dist_sq = np.min(dist_sq)
    collision_penalty = beta * np.exp(-min_dist_sq / 0.001)
    
    return float(log_sum_exp + 0.1 * hull_penalty + collision_penalty)

def _optimize_thomson(init_pts, max_iter=500):
    """Optimize all points using Thomson energy with L-BFGS-B."""
    n = len(init_pts)
    init_flat = init_pts.flatten()
    
    try:
        result = minimize(
            _thomson_objective,
            init_flat,
            method='L-BFGS-B',
            bounds=[(0.05, 0.95)] * (2 * n),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return init_pts

def _soft_min_objective(pts_flat, alpha=50):
    """
    Direct soft-min objective: maximize min triangle area / hull area.
    All 13 points are free to move.
    """
    n = len(pts_flat) // 2
    pts = pts_flat.reshape(-1, 2)
    
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Soft minimum
        max_neg = np.max(-alpha * areas)
        log_sum_exp = np.log(np.sum(np.exp(-alpha * areas - max_neg))) + max_neg
        soft_min_area = -log_sum_exp / alpha
        
        return -(soft_min_area / ha)
    except Exception:
        return 1e6

def _optimize_soft_min(init_pts, max_iter=300):
    """Optimize all points using soft-min objective."""
    n = len(init_pts)
    init_flat = init_pts.flatten()
    
    try:
        result = minimize(
            _soft_min_objective,
            init_flat,
            method='L-BFGS-B',
            bounds=[(0.01, 0.99)] * (2 * n),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return init_pts

def _coordinate_descent_polish(points0, max_sweeps=30):
    """Fine-grained coordinate descent polish on all points."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    step_sizes = [0.003, 0.001, 0.0005, 0.0002]
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local = best_score
            
            for step in step_sizes:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step*0.707, step*0.707), (-step*0.707, step*0.707),
                               (step*0.707, -step*0.707), (-step*0.707, -step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_x, best_y = nx, ny
                        improved = True
            
            pts[i] = [best_x, best_y]
        
        cur = _score(pts)
        if cur > best_score + 1e-12:
            best_score = cur
            best_pts = pts.copy()
        
        if not improved:
            break
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Start from regular 13-gon, optimize with Thomson energy
    if time.time() < _DEADLINE - 100:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 0.45
        center = (0.5, 0.5)
        init1 = np.column_stack([center[0] + radius * np.cos(angles),
                                  center[1] + radius * np.sin(angles)])
        
        optimized1 = _optimize_thomson(init1, max_iter=300)
        s1 = _score(optimized1)
        if s1 > best_score:
            best_score = s1
            best_points = optimized1.copy()
    
    # Strategy 2: Start from regular 12-gon + center, optimize with soft-min
    if time.time() < _DEADLINE - 80:
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radius = 0.45
        center = (0.5, 0.5)
        hull_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                     center[1] + radius * np.sin(angles)])
        init2 = np.vstack([hull_pts, np.array([center])])
        
        optimized2 = _optimize_soft_min(init2, max_iter=200)
        s2 = _score(optimized2)
        if s2 > best_score:
            best_score = s2
            best_points = optimized2.copy()
    
    # Strategy 3: Start from two concentric rings (7 outer, 6 inner)
    if time.time() < _DEADLINE - 60:
        angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        r_out = 0.45
        r_in = 0.2
        center = (0.5, 0.5)
        outer = np.column_stack([center[0] + r_out * np.cos(angles_out),
                                  center[1] + r_out * np.sin(angles_out)])
        inner = np.column_stack([center[0] + r_in * np.cos(angles_in),
                                  center[1] + r_in * np.sin(angles_in)])
        init3 = np.vstack([outer, inner])
        
        optimized3 = _optimize_soft_min(init3, max_iter=200)
        s3 = _score(optimized3)
        if s3 > best_score:
            best_score = s3
            best_points = optimized3.copy()
    
    # Strategy 4: Perturbed regular 13-gon with Thomson
    if time.time() < _DEADLINE - 50:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = 0.45 + 0.05 * np.sin(3 * angles)
        center = (0.5, 0.5)
        init4 = np.column_stack([center[0] + radii * np.cos(angles),
                                  center[1] + radii * np.sin(angles)])
        
        optimized4 = _optimize_thomson(init4, max_iter=200)
        s4 = _score(optimized4)
        if s4 > best_score:
            best_score = s4
            best_points = optimized4.copy()
    
    # Final: coordinate descent polish on best result
    if best_points is not None and time.time() < _DEADLINE - 10:
        polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=20)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 0.45
        center = (0.5, 0.5)
        best_points = np.column_stack([center[0] + radius * np.cos(angles),
                                        center[1] + radius * np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
