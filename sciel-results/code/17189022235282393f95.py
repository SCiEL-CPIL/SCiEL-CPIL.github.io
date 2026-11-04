"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return 0.0
    if len(pts) == 2:
        return 0.0
    
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]
    return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        return 0.0
    return min_ta / hull_a

def _build_points(params):
    """
    params: 7-dim array of radial offsets.
    Each of 7 heptagon vertices is split into 2 points at radius 1+offset and 1-offset.
    Returns 14 points.
    """
    offsets = params
    points = []
    for i in range(7):
        r1 = 1.0 + offsets[i]
        r2 = 1.0 - offsets[i]
        r1 = max(r1, 0.05)
        r2 = max(r2, 0.05)
        angle = i * 2 * np.pi / 7
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        points.append([r1 * cos_a, r1 * sin_a])
        points.append([r2 * cos_a, r2 * sin_a])
    return np.array(points)

def _negative_score(params):
    """Objective: minimize negative score."""
    points = _build_points(params)
    score = _score(points)
    return -score

def construct():
    """
    Construct 14 points by optimizing 7 radial offsets for a regular heptagon
    where each vertex is split into two radially displaced points.
    """
    np.random.seed(123)
    
    # Define bounds for each offset: keep radii positive and reasonable
    bounds = [(-0.4, 0.4)] * 7
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        _negative_score, 
        bounds,
        seed=42,
        maxiter=200,
        tol=1e-8,
        popsize=15,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True
    )
    
    best_params = result.x
    best_score = -result.fun
    
    # Additional local refinement with Nelder-Mead
    try:
        res = minimize(_negative_score, best_params, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if -res.fun > best_score:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    # Try a few more random starts with local optimization
    for trial in range(20):
        random_params = np.random.uniform(-0.3, 0.3, 7)
        try:
            res = minimize(_negative_score, random_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
            if -res.fun > best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    points = _build_points(best_params)
    
    # Verify validity
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        # Fallback
        points = []
        for i in range(7):
            angle = i * 2 * np.pi / 7
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            points.append([1.1 * cos_a, 1.1 * sin_a])
            points.append([0.9 * cos_a, 0.9 * sin_a])
        points = np.array(points)
    
    return points
# EVOLVE-BLOCK-END
