import numpy as np
import time

N = 14

def _min_triangle_area(points):
    """Compute minimum area of any triangle from triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    mta = _min_triangle_area(points)
    cha = _convex_hull_area(points)
    if cha < 1e-12:
        return 0.0
    return mta / cha

def construct():
    deadline = time.time() + 120  # safety margin before 150s kill
    
    # Start with regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Local search: perturb points and accept improvements
    rng = np.random.RandomState(42)
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate perturbation
        perturbation = rng.randn(N, 2) * step_size
        new_points = best_points + perturbation
        
        new_score = _score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            no_improve_count = 0
            # Slightly reduce step size on improvement
            step_size *= 0.95
        else:
            no_improve_count += 1
            # Slowly reduce step size
            step_size *= 0.999
    
    return best_points
