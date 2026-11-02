"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Sort by x, then y
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    # Build lower hull
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in sorted_pts[::-1]:
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    # Compute area using shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) / 2.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _heuristic_init():
    """Initialize with a symmetric configuration that approximates optimal Heilbronn."""
    # Start with points on a regular 13-gon plus some interior points
    # Actually, let's use a grid-like pattern perturbed slightly
    # Better: use a known good starting point - regular polygon with center
    points = []
    # 12 points on a circle, 1 at center
    for i in range(12):
        angle = 2 * np.pi * i / 12
        points.append([np.cos(angle), np.sin(angle)])
    points.append([0.0, 0.0])
    return np.array(points)

def construct():
    """
    Construct 13 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    # Start with a heuristic initialization
    x0 = _heuristic_init()
    
    # Normalize to unit area
    hull_area = _convex_hull_area(x0)
    if hull_area > 0:
        x0 = x0 / np.sqrt(hull_area)
    
    # Use Nelder-Mead optimization to maximize the score
    def neg_score(x_flat):
        points = x_flat.reshape(-1, 2)
        return -_score(points)
    
    # Run optimization
    result = minimize(
        neg_score,
        x0.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    best_points = result.x.reshape(-1, 2)
    best_score = -result.fun
    
    # Also try a few random restarts to improve robustness
    for seed in [42, 123, 999]:
        rng = np.random.RandomState(seed)
        x_rand = rng.rand(N, 2)
        # Normalize
        hull_area_rand = _convex_hull_area(x_rand)
        if hull_area_rand > 0:
            x_rand = x_rand / np.sqrt(hull_area_rand)
        
        result_rand = minimize(
            neg_score,
            x_rand.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        rand_points = result_rand.x.reshape(-1, 2)
        rand_score = -result_rand.fun
        
        if rand_score > best_score:
            best_points = rand_points
            best_score = rand_score
    
    return best_points
# EVOLVE-BLOCK-END

