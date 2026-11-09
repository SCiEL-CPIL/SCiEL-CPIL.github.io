"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    Returns: minimum triangle area (positive value).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            # Vectorized over k > j
            if j + 1 >= n:
                continue
            pk = points[j+1:]
            # Cross product of (pj - pi) and (pk - pi)
            v1 = pj - pi
            v2 = pk - pi
            # Area = 0.5 * |cross(v1, v2)|
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    Points: (N, 2) array.
    Returns: convex hull area.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0


def _score(points):
    """
    Score = min triangle area / convex hull area.
    Higher is better.
    """
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(points)
    return min_ta / hull_a


def _negative_score(points_flat):
    """
    Objective function for minimizer: minimize negative score.
    points_flat: 1D array of length 2*N.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points of a regular n-gon centered at 'center' with given radius.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))


def _perturb(points, sigma, rng):
    """
    Add Gaussian noise to points.
    """
    noise = rng.normal(0, sigma, size=points.shape)
    return points + noise


def _local_optimize(points, max_iter=50, tol=1e-10):
    """
    Use Nelder-Mead to locally optimize the score.
    Returns optimized points and score.
    """
    x0 = points.flatten()
    res = minimize(_negative_score, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    opt_points = res.x.reshape(N, 2)
    score = _score(opt_points)
    return opt_points, score


def construct():
    """
    Construct 14 points to maximize the minimum triangle area divided by convex hull area.
    Strategy:
    1. Start with a regular 14-gon.
    2. Perform local optimization using Nelder-Mead.
    3. Try multiple random restarts with perturbed regular polygons and other configurations.
    4. Keep track of the best solution found within the time limit.
    """
    start_time = time.time()
    time_limit = 115  # Safety margin before 120s limit
    
    best_points = None
    best_score = -1.0
    
    # Helper to update best
    def update_best(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # 1. Start with regular 14-gon
    rng = np.random.RandomState(42)
    
    # Initial regular polygon
    pts = _regular_polygon(N, radius=1.0)
    sc = _score(pts)
    update_best(pts, sc)
    
    # Local optimize the regular polygon
    if time.time() - start_time < time_limit:
        opt_pts, opt_sc = _local_optimize(pts, max_iter=100)
        update_best(opt_pts, opt_sc)
    
    # 2. Random restarts
    # Try perturbing the regular polygon
    for i in range(50):
        if time.time() - start_time >= time_limit:
            break
        
        # Different perturbation strategies
        if i % 3 == 0:
            # Small perturbation of regular polygon
            base = _regular_polygon(N, radius=1.0)
            sigma = 0.05
            pts = _perturb(base, sigma, rng)
        elif i % 3 == 1:
            # Medium perturbation
            base = _regular_polygon(N, radius=1.0)
            sigma = 0.1
            pts = _perturb(base, sigma, rng)
        else:
            # Random points in unit circle
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.5, 1.0, N)
            pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        sc = _score(pts)
        update_best(pts, sc)
        
        # Local optimize if score is promising
        if sc > best_score * 0.9 and time.time() - start_time < time_limit - 5:
            opt_pts, opt_sc = _local_optimize(pts, max_iter=50)
            update_best(opt_pts, opt_sc)
    
    # 3. Try some structured configurations
    # Concentric polygons
    for r1, r2 in [(0.5, 1.0), (0.6, 1.0), (0.7, 1.0)]:
        if time.time() - start_time >= time_limit:
            break
        
        # 7 points on inner circle, 7 on outer circle, offset by half step
        angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_outer = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False)
        
        inner_pts = np.column_stack((r1 * np.cos(angles_inner), r1 * np.sin(angles_inner)))
        outer_pts = np.column_stack((r2 * np.cos(angles_outer), r2 * np.sin(angles_outer)))
        
        pts = np.vstack((inner_pts, outer_pts))
        sc = _score(pts)
        update_best(pts, sc)
        
        if time.time() - start_time < time_limit - 5:
            opt_pts, opt_sc = _local_optimize(pts, max_iter=50)
            update_best(opt_pts, opt_sc)
    
    # 4. Final polish: run more local optimization on the best point
    if best_points is not None and time.time() - start_time < time_limit - 5:
        opt_pts, opt_sc = _local_optimize(best_points, max_iter=200, tol=1e-12)
        update_best(opt_pts, opt_sc)
    
    if best_points is None:
        # Fallback: return regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
