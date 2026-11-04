"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_start_time = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                 (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _superellipse_points(shape_exponent, aspect_ratio, n_points):
    """
    Place n_points on a superellipse |x/a|^p + |y/b|^p = 1,
    distributed uniformly by arc length.
    
    shape_exponent: p (Lamé exponent)
    aspect_ratio: a/b ratio
    """
    p = shape_exponent
    a = np.sqrt(aspect_ratio)  # semi-axis a
    b = 1.0 / np.sqrt(aspect_ratio)  # semi-axis b, so a*b = 1 (unit area approx)
    
    # Sample many points on the superellipse parameterized by angle
    # For superellipse: x = a * sign(cos(t)) * |cos(t)|^(2/p), y = b * sign(sin(t)) * |sin(t)|^(2/p)
    t_samples = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    cos_t = np.cos(t_samples)
    sin_t = np.sin(t_samples)
    
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / p)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / p)
    
    # Compute cumulative arc length
    dx = np.diff(x, append=[x[0]])
    dy = np.diff(y, append=[y[0]])
    ds = np.sqrt(dx**2 + dy**2)
    cumulative = np.cumsum(ds)
    total_length = cumulative[-1]
    
    # Uniform spacing by arc length
    target_lengths = np.linspace(0, total_length, n_points, endpoint=False)
    
    # Interpolate to find points at target arc lengths
    idx = np.searchsorted(cumulative, target_lengths)
    idx = np.clip(idx, 0, len(cumulative) - 1)
    
    # Linear interpolation for accuracy
    pts = np.zeros((n_points, 2))
    for i in range(n_points):
        tl = target_lengths[i]
        j = idx[i]
        if j >= len(cumulative) - 1:
            j = len(cumulative) - 2
        if cumulative[j] == cumulative[j + 1]:
            pts[i] = [x[j], y[j]]
        else:
            frac = (tl - cumulative[j]) / (cumulative[j + 1] - cumulative[j])
            frac = np.clip(frac, 0, 1)
            pts[i, 0] = x[j] + frac * (x[(j + 1) % len(x)] - x[j])
            pts[i, 1] = y[j] + frac * (y[(j + 1) % len(y)] - y[j])
    
    return pts


def _objective(params):
    """Negative score for minimization."""
    if time.time() > DEADLINE:
        return 1e10
    p, aspect = params
    if p < 1.0 or p > 100.0 or aspect < 0.1 or aspect > 10.0:
        return 1e10
    try:
        pts = _superellipse_points(p, aspect, N)
        s = _score(pts)
        return -s
    except Exception:
        return 1e10


def construct():
    global _start_time
    _start_time = time.time()
    
    best_score = 0.0
    best_points = None
    
    # Initial random points as fallback
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.rand(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Grid search over shape parameters
    p_values = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]
    aspect_values = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    for p in p_values:
        if time.time() > DEADLINE - 5:
            break
        for aspect in aspect_values:
            if time.time() > DEADLINE - 5:
                break
            try:
                pts = _superellipse_points(p, aspect, N)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Local optimization around best found parameters
    # Find which grid point gave the best score
    best_p, best_aspect = 2.0, 1.0
    for p in p_values:
        for aspect in aspect_values:
            if time.time() > DEADLINE - 10:
                break
            try:
                pts = _superellipse_points(p, aspect, N)
                s = _score(pts)
                if s >= best_score - 1e-10:
                    best_p, best_aspect = p, aspect
            except Exception:
                pass
        if time.time() > DEADLINE - 10:
            break
    
    # Local refinement
    if time.time() < DEADLINE - 10:
        x0 = [best_p, best_aspect]
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < -best_score:
                p_opt, aspect_opt = result.x
                if 1.0 < p_opt < 100.0 and 0.1 < aspect_opt < 10.0:
                    pts = _superellipse_points(p_opt, aspect_opt, N)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
        except Exception:
            pass
    
    # Additional random restarts in parameter space
    if time.time() < DEADLINE - 5 and best_points is not None:
        rng2 = np.random.RandomState(123)
        for _ in range(20):
            if time.time() > DEADLINE - 3:
                break
            p_try = 1.5 + rng2.rand() * 10.0
            aspect_try = 0.5 + rng2.rand() * 2.0
            try:
                pts = _superellipse_points(p_try, aspect_try, N)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback to random
        return np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_points
# EVOLVE-BLOCK-END
