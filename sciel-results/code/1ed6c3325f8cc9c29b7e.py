"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: shoelace formula on points (assumes convex position)
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
        return abs(area) / 2.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(points, temp=0.01):
    """Smooth approximation of -log(sum(exp(-area/temp))) for optimization.
    Returns negative log-sum-exp of triangle areas (to be minimized)."""
    areas = _all_triangle_areas(points)
    # Log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - min_a) / temp))) + (-min_a / temp)
    # We want to maximize min area, so minimize -soft_min
    return -(-min_a / temp - log_sum)  # = min_a/temp + log_sum ... let me redo this


def _soft_min_neg(points, temp=0.01):
    """Return value to minimize: -(soft minimum of triangle areas).
    soft_min = -temp * log(sum(exp(-area/temp)))
    We want to maximize soft_min, so minimize -soft_min = temp * log(sum(exp(-area/temp)))
    """
    areas = _all_triangle_areas(points)
    # For numerical stability
    min_a = np.min(areas)
    shifted = -(areas - min_a) / temp
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = -temp * (log_sum + min_a / temp)  # = -temp * log_sum - min_a
    # Wait: soft_min = -temp * log(sum(exp(-a/temp)))
    # sum(exp(-a/temp)) = sum(exp(-(a-min_a)/temp) * exp(-min_a/temp))
    #                   = exp(-min_a/temp) * sum(exp(-(a-min_a)/temp))
    # log(sum) = -min_a/temp + log(sum(exp(-(a-min_a)/temp)))
    # soft_min = -temp * (-min_a/temp + log_sum_shifted) = min_a - temp * log_sum_shifted
    log_sum_shifted = np.log(np.sum(np.exp(-(areas - min_a) / temp)))
    soft_min = min_a - temp * log_sum_shifted
    return -soft_min  # minimize this


def _radial_to_points(radial_params, base_radius=1.0):
    """Convert 13 radial parameters to 2D points on a perturbed regular 13-gon."""
    n = len(radial_params)
    angles = 2 * np.pi * np.arange(n) / n
    radii = base_radius + radial_params
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _optimize_radial(initial_radial, max_iter=200):
    """Optimize radial parameters using L-BFGS-B."""
    def objective(radial):
        points = _radial_to_points(radial)
        return _soft_min_neg(points, temp=0.005)
    
    bounds = [(-0.5, 0.5)] * N  # radial perturbation bounds
    result = minimize(objective, initial_radial, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x


def _optimize_2d(points, max_iter=100):
    """Polish with full 2D L-BFGS-B."""
    def objective_flat(flat):
        pts = flat.reshape(-1, 2)
        return _soft_min_neg(pts, temp=0.005)
    
    result = minimize(objective_flat, points.flatten(), method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x.reshape(-1, 2)


def _worst_triangle_optimization(points, max_iter=50):
    """Targeted optimization focusing on the worst (smallest) triangle."""
    def objective(flat):
        pts = flat.reshape(-1, 2)
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        # Return negative of the minimum area (to maximize it)
        # But we need smoothness, so use soft min with small temp
        return _soft_min_neg(pts, temp=0.001)
    
    result = minimize(objective, points.flatten(), method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x.reshape(-1, 2)


def construct():
    global _START_TIME
    best_points = None
    best_score = 0.0
    
    # Start with regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    base_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Evaluate base
    s = _score(base_points)
    best_points = base_points.copy()
    best_score = s
    
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_points
    
    # Phase 1: Radial optimization from regular polygon
    initial_radial = np.zeros(N)
    try:
        optimized_radial = _optimize_radial(initial_radial, max_iter=150)
        points_r = _radial_to_points(optimized_radial)
        s_r = _score(points_r)
        if s_r > best_score:
            best_score = s_r
            best_points = points_r.copy()
    except Exception:
        pass
    
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_points
    
    # Phase 2: Full 2D polish
    try:
        points_2d = _optimize_2d(best_points, max_iter=80)
        s_2d = _score(points_2d)
        if s_2d > best_score:
            best_score = s_2d
            best_points = points_2d.copy()
    except Exception:
        pass
    
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_points
    
    # Phase 3: Worst-triangle targeted optimization
    try:
        points_wt = _worst_triangle_optimization(best_points, max_iter=60)
        s_wt = _score(points_wt)
        if s_wt > best_score:
            best_score = s_wt
            best_points = points_wt.copy()
    except Exception:
        pass
    
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_points
    
    # Phase 4: Another 2D polish from best
    try:
        points_final = _optimize_2d(best_points, max_iter=60)
        s_final = _score(points_final)
        if s_final > best_score:
            best_score = s_final
            best_points = points_final.copy()
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
