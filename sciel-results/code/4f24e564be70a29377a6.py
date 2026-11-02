"""Heilbronn 11 triangles: Differential Evolution global search with local polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _project_to_triangle(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    """
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


def _soft_min_area(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _is_inside_triangle(x, y):
    """Check if a point is inside the triangle."""
    if y < 0 or y > S3H:
        return False
    x_min = y / S3
    x_max = 1 - y / S3
    return x_min <= x <= x_max


def _penalized_objective(x):
    """
    Objective function for Differential Evolution.
    Returns negative min area.
    Includes a penalty for points outside the triangle to guide DE back inside.
    """
    pts = x.reshape(-1, 2)
    
    # Check validity and apply penalty
    penalty = 0.0
    for i in range(N):
        x, y = pts[i]
        if not _is_inside_triangle(x, y):
            # Calculate distance to the triangle (approximate) and penalize
            # Simple penalty: large value
            penalty += 1.0 
            
    if penalty > 0:
        # If any point is outside, return a very bad score
        # This helps DE avoid the outside regions
        return 10.0 + penalty

    # Project to be safe (handles floating point errors)
    pts = _project_to_triangle(pts)
    area = _min_triangle_area(pts)
    return -area


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Phase 1: Differential Evolution
    # Bounds: Bounding box of the triangle
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    
    # DE parameters
    # Popsize 30, maxiter 100
    # This might be slow, so we need to be careful.
    # 30 * 100 * 22 dims? No, DE evaluates popsize * (1 + 1) per iteration roughly.
    # 30 * 100 = 3000 evaluations.
    # Each eval is ~165 cross products. Very fast.
    # 3000 evals should take < 1 second.
    
    try:
        if _elapsed_ok(10):
            # Use a seed for reproducibility in DE
            de_result = differential_evolution(
                _penalized_objective,
                bounds,
                maxiter=100,
                popsize=30,
                tol=1e-6,
                seed=42,
                polish=False, # We will do our own polish
                init='latinhypercube'
            )
            if de_result.success or de_result.fun < 0: # fun < 0 means valid area found
                de_pts = de_result.x.reshape(-1, 2)
                _update_best(de_pts)
    except Exception as e:
        pass

    # Phase 2: Local Polish with Nelder-Mead on the best candidate
    if best_pts is not None and _elapsed_ok(10):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            # Add penalty for being outside to keep NM inside
            penalty = 0
            for i in range(N):
                if not _is_inside_triangle(p[i, 0], p[i, 1]):
                    penalty += 1.0
            if penalty > 0:
                return 10.0 + penalty
            return -_min_triangle_area(p)
        
        try:
            res = minimize(
                nm_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    # Phase 3: Multi-start with Random Initialization if time permits
    if _elapsed_ok(30):
        for _ in range(5):
            # Random initialization inside triangle
            pts = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1:
                        x = u + 0.5 * v
                        y = S3H * v
                        pts.append([x, y])
                        break
            pts = np.array(pts)
            
            # Quick local polish
            def nm_obj_rand(x):
                p = x.reshape(-1, 2)
                p = _project_to_triangle(p)
                penalty = 0
                for i in range(N):
                    if not _is_inside_triangle(p[i, 0], p[i, 1]):
                        penalty += 1.0
                if penalty > 0:
                    return 10.0 + penalty
                return -_min_triangle_area(p)
            
            try:
                res = minimize(
                    nm_obj_rand,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                p = res.x.reshape(-1, 2)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    if best_pts is None:
        # Absolute fallback
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        best_pts = _project_to_triangle(pts)

    return best_pts
