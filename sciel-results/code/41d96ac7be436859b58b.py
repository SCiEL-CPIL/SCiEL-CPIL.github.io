"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-12

def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return -1e6
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return -1e6
    return min_area / hull_area

def _negative_score(points):
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start from regular 14-gon
    x0 = _regular_polygon(N)
    
    best_points = x0.copy()
    best_score = _score(best_points)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _negative_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.x is not None:
            candidate = result.x.reshape(N, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_points = candidate
                best_score = cand_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # Try a few more local optimizations from perturbed starts
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Perturb the regular polygon
        perturb = rng.normal(0, 0.05, (N, 2))
        x_trial = x0 + perturb
        
        try:
            result = minimize(
                _negative_score,
                x_trial,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.x is not None:
                candidate = result.x.reshape(N, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_points = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
