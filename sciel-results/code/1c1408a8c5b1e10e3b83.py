"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized)."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        area = 0.5 * cross
        if area < min_area:
            min_area = area
    return min_area

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = len(pts)
    areas = []
    for i, j, k in combinations(range(n), 3):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        areas.append(0.5 * cross)
    return np.array(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _soft_min(areas, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    # min(a) ≈ -(1/beta) * log(sum(exp(-beta * a)))
    # Clip to avoid overflow
    areas_clipped = np.clip(areas, 1e-12, None)
    neg_beta_a = -beta * areas_clipped
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    return -log_sum / beta

def _smooth_objective(x_flat, beta=50.0):
    """Smooth objective: negative soft-min of triangle areas."""
    pts = x_flat.reshape(N, 2)
    if not _check_inside(pts, eps=1e-6):
        return 1e10
    areas = _all_triangle_areas(pts)
    return -_soft_min(areas, beta)

def _epigraph_objective(x_flat, beta=50.0):
    """
    Epigraph formulation: variables are (x_0..x_20, t) where t is the min area.
    Minimize -t + penalty for areas < t.
    """
    n_vars = N * 2 + 1
    pts = x_flat[:N*2].reshape(N, 2)
    t = x_flat[N*2]
    
    if not _check_inside(pts, eps=1e-6):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    # Penalty for areas below t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + 100.0 * penalty

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _structured_guess(rng):
    """Generate a structured initial guess using vertices, edge midpoints, and center."""
    # Triangle vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, S3H])
    
    # Edge midpoints
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    
    # Centroid
    centroid = (v0 + v1 + v2) / 3
    
    # Start with these 7 points, add 4 more randomly
    pts = [v0, v1, v2, m01, m12, m20, centroid]
    for _ in range(4):
        pts.append(_random_point_in_triangle(rng))
    
    return np.array(pts)

def _initial_guess(rng, structured=False):
    """Generate an initial set of 11 points."""
    if structured:
        return _structured_guess(rng)
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def _polish(pts, start_time):
    """High-precision polish pass using smooth objective with tight tolerances."""
    x0 = pts.flatten()
    
    # Use smooth objective with high beta for near-exact min
    try:
        result = minimize(
            lambda x: _smooth_objective(x, beta=100.0),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
    except Exception:
        return pts
    
    opt_pts = result.x.reshape(N, 2)
    opt_pts = _project_to_triangle(opt_pts)
    
    if _check_inside(opt_pts):
        return opt_pts
    return pts

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Phase 1: Multiple random and structured starts with smooth optimization
    max_starts = 40
    
    for trial in range(max_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Alternate between random and structured starts
        use_structured = (trial % 3 == 0)
        pts = _initial_guess(rng, structured=use_structured)
        x0 = pts.flatten()
        
        # Run optimization with smooth objective
        try:
            result = minimize(
                lambda x: _smooth_objective(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000}
            )
        except Exception:
            continue
        
        opt_pts = result.x.reshape(N, 2)
        opt_pts = _project_to_triangle(opt_pts)
        
        if _check_inside(opt_pts):
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
        
        # Also try the initial points
        init_area = _min_triangle_area(pts)
        if init_area > best_area:
            best_area = init_area
            best_pts = pts.copy()
    
    # Phase 2: Polish the best result found
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        polished = _polish(best_pts, start_time)
        polished_area = _min_triangle_area(polished)
        if polished_area > best_area:
            best_area = polished_area
            best_pts = polished
    
    # Phase 3: Try epigraph formulation from best point
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        x0 = np.concatenate([best_pts.flatten(), [best_area]])
        try:
            result = minimize(
                lambda x: _epigraph_objective(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000}
            )
            opt_pts = result.x[:N*2].reshape(N, 2)
            opt_pts = _project_to_triangle(opt_pts)
            if _check_inside(opt_pts):
                area = _min_triangle_area(opt_pts)
                if area > best_area:
                    best_area = area
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        pts = _initial_guess(rng)
        return pts
    
    return best_pts
