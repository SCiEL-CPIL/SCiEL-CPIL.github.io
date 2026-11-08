"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import sys

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin seconds

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace on convex hull points
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            area = 0.0
            n = len(hull.vertices)
            for i in range(n):
                j = (i + 1) % n
                area += points[hull.vertices[i]][0] * points[hull.vertices[j]][1]
                area -= points[hull.vertices[j]][0] * points[hull.vertices[i]][1]
            return abs(area) / 2.0
        except Exception:
            return 1.0

def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices i, j, k
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0 or not np.isfinite(hull_area):
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _fourier_to_points(params):
    """
    Convert Fourier series parameters to point coordinates.
    params: [r0, a1, b1, a2, b2, ..., a_m, b_m] where m is number of harmonics.
    r0 is the base radius, a_k and b_k are Fourier coefficients for harmonic k.
    
    We use a regular 13-gon as the base shape and deform it with low-order harmonics.
    """
    n = N
    # params layout: [r0, a1, b1, a2, b2, a3, b3, ...]
    r0 = params[0]
    num_harmonics = (len(params) - 1) // 2
    
    angles = 2 * np.pi * np.arange(n) / n
    radii = np.zeros(n)
    for i in range(n):
        r = r0
        for k in range(1, num_harmonics + 1):
            a_k = params[1 + 2*(k-1)]
            b_k = params[2 + 2*(k-1)]
            r += a_k * np.cos(k * angles[i]) + b_k * np.sin(k * angles[i])
        radii[i] = r
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, subtract max
    neg_areas = -beta * areas
    max_neg = np.max(neg_areas)
    log_sum = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
    soft_min = -log_sum / beta
    return soft_min

def _soft_score(params, beta=100.0):
    """
    Smooth surrogate for the score objective.
    We want to maximize min_triangle_area / hull_area.
    This is equivalent to minimizing -min_triangle_area / hull_area.
    """
    points = _fourier_to_points(params)
    # Check if points are valid (no NaN, etc.)
    if not np.all(np.isfinite(points)):
        return 1e10
    hull_area = _convex_hull_area(points)
    if hull_area <= 0 or not np.isfinite(hull_area):
        return 1e10
    soft_min_area = _soft_min_triangle_area(points, beta)
    # We want to maximize soft_min_area / hull_area, so minimize the negative
    return -soft_min_area / hull_area

def _exact_objective(params):
    """Exact objective for local refinement: minimize -score."""
    points = _fourier_to_points(params)
    if not np.all(np.isfinite(points)):
        return 1e10
    score = _score(points)
    if not np.isfinite(score):
        return 1e10
    return -score

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon with given radius."""
    angles = 2 * np.pi * np.arange(n) / n
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 150.0
    _MARGIN = 5.0
    
    best_points = _regular_polygon(N, radius=1.0)
    best_score = _score(best_points)
    
    # Fourier parameterization: r0 + harmonics 1 through 3
    # params = [r0, a1, b1, a2, b2, a3, b3] -> 7 variables
    num_harmonics = 3
    n_params = 1 + 2 * num_harmonics
    
    # Bounds for parameters
    # r0 should be positive, around 1.0
    # Fourier coefficients should be small to maintain convexity
    bounds = [(0.5, 2.0)] + [(-0.3, 0.3)] * (2 * num_harmonics)
    
    # Initial guess: regular polygon corresponds to r0=1, all harmonics=0
    x0 = np.array([1.0] + [0.0] * (2 * num_harmonics))
    
    # Use differential evolution to find a good starting point
    if _time_left() > 30:
        try:
            result = differential_evolution(
                _soft_score,
                bounds=bounds,
                args=(50.0,),
                seed=42,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='sobol'
            )
            if result.success or result.fun < 1e9:
                x0 = result.x
                # Update best
                pts = _fourier_to_points(x0)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead on the exact objective
    if _time_left() > 10:
        try:
            result = minimize(
                _exact_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < 1e9:
                x0 = result.x
                pts = _fourier_to_points(x0)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts
        except Exception:
            pass
    
    # Final high-precision polish with Powell
    if _time_left() > 5:
        try:
            result = minimize(
                _exact_objective,
                x0,
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if result.fun < 1e9:
                x0 = result.x
                pts = _fourier_to_points(x0)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts
        except Exception:
            pass
    
    # Coordinate descent refinement
    if _time_left() > 2:
        try:
            current = x0.copy()
            for _ in range(50):
                if _time_left() < 1:
                    break
                improved = False
                for i in range(len(current)):
                    best_val = _exact_objective(current)
                    best_x = current[i]
                    lo, hi = bounds[i]
                    # Try a few steps around current value
                    for frac in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                        test = current.copy()
                        test[i] = current[i] + frac * (hi - lo)
                        if test[i] < lo or test[i] > hi:
                            continue
                        val = _exact_objective(test)
                        if val < best_val - 1e-14:
                            best_val = val
                            best_x = test[i]
                    if best_x != current[i]:
                        current[i] = best_x
                        improved = True
                if not improved:
                    break
            pts = _fourier_to_points(current)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
