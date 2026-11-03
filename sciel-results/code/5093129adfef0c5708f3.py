"""Heilbronn-11: Active-Set Coordinate Descent with Delaunay-guided restarts."""
import numpy as np
import time
import itertools
from scipy.spatial import Delaunay

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 140.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2)."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v."""
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _active_set_polish(pts_cart, max_iters=50):
    """
    Active-set coordinate descent:
    1. Find the critical (minimum area) triangle.
    2. For each point in that triangle, try to move it to increase the minimum area.
    3. Use a 1D line search along a direction that increases the critical area.
    """
    current_min_area = _vectorized_min_area_cart(pts_cart)
    pts = pts_cart.copy()
    
    for iteration in range(max_iters):
        # Find all triangles with area close to the minimum
        pi = pts[_T_I]
        pj = pts[_T_J]
        pk = pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        min_area = np.min(areas)
        if min_area <= current_min_area + 1e-12:
            break
        
        # Find indices of triangles with minimum area
        min_idx = np.where(areas <= min_area + 1e-10)[0]
        
        # Collect unique points involved in critical triangles
        critical_points = set()
        for idx in min_idx:
            critical_points.add(_T_I[idx])
            critical_points.add(_T_J[idx])
            critical_points.add(_T_K[idx])
        
        # Try to move each critical point
        improved = False
        for p_idx in sorted(critical_points):
            # Try moving point p_idx in several directions
            # Direction 1: away from the centroid of the triangle
            for trial in range(20):
                # Random perturbation
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(0.001, 0.05)
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
                
                new_pts = pts.copy()
                new_pts[p_idx, 0] += dx
                new_pts[p_idx, 1] += dy
                new_pts = _project_cartesian(new_pts)
                
                new_min_area = _vectorized_min_area_cart(new_pts)
                if new_min_area > current_min_area + 1e-10:
                    pts = new_pts
                    current_min_area = new_min_area
                    improved = True
                    break
        
        if not improved:
            break
    
    return pts, current_min_area


def _delaunay_guided_init(rng):
    """
    Initialize points using a Delaunay-inspired strategy:
    Start with a coarse grid, then refine using Delaunay triangulation properties.
    """
    # Start with a structured grid
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    
    # Add random jitter
    pts += rng.normal(0, 0.02, size=pts.shape)
    pts = _project_cartesian(pts)
    
    return pts


def _local_search(pts_cart, rng, max_evals=2000):
    """
    Local search with adaptive step size.
    """
    current_min_area = _vectorized_min_area_cart(pts_cart)
    pts = pts_cart.copy()
    step_size = 0.05
    evals = 0
    
    while evals < max_evals:
        # Pick a random point
        p_idx = rng.integers(0, N)
        
        # Generate a random direction
        angle = rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_pts = pts.copy()
        new_pts[p_idx, 0] += dx
        new_pts[p_idx, 1] += dy
        new_pts = _project_cartesian(new_pts)
        
        new_min_area = _vectorized_min_area_cart(new_pts)
        evals += 1
        
        if new_min_area > current_min_area:
            pts = new_pts
            current_min_area = new_min_area
            step_size *= 1.1  # Increase step if improving
        else:
            step_size *= 0.9  # Decrease step if not improving
            step_size = max(step_size, 0.001)
    
    return pts, current_min_area


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Phase 1: Multiple initializations with local search
    n_starts = 8
    for start_idx in range(n_starts):
        if not _elapsed_ok(30):
            break
        
        if start_idx == 0:
            # Use structured initialization
            pts = _delaunay_guided_init(rng)
        else:
            # Random initialization with some structure
            pts = np.zeros((N, 2))
            # Place corners and center first
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2] = [0.5, S3H]
            pts[3] = [0.5, 0.0]
            pts[4] = [0.75, S3H / 2]
            pts[5] = [0.25, S3H / 2]
            # Random for the rest
            for i in range(6, N):
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1 - u)
                w = 1 - u - v
                x = v + 0.5 * w
                y = w * S3H
                pts[i] = [x, y]
            pts += rng.normal(0, 0.01, size=pts.shape)
            pts = _project_cartesian(pts)
        
        _update_best(pts)
        
        # Local search
        if _elapsed_ok(20):
            pts_ls, area_ls = _local_search(pts, rng, max_evals=500)
            _update_best(pts_ls)
    
    # Phase 2: Active-set polish on the best configuration
    if best_pts is not None and _elapsed_ok(60):
        pts_polished, area_polished = _active_set_polish(best_pts, max_iters=100)
        _update_best(pts_polished)
    
    # Phase 3: More local search with smaller steps
    if best_pts is not None and _elapsed_ok(30):
        pts_fine, area_fine = _local_search(best_pts, rng, max_evals=3000)
        _update_best(pts_fine)
    
    # Phase 4: Final active-set polish
    if best_pts is not None and _elapsed_ok(20):
        pts_final, area_final = _active_set_polish(best_pts, max_iters=200)
        _update_best(pts_final)
    
    if best_pts is None:
        # Fallback: return a valid configuration
        return _delaunay_guided_init(rng)
    
    return best_pts
