import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = pts[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace on sorted points
        idx = np.argsort(pts[:, 0])
        p = pts[idx]
        n = len(p)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += p[i][0] * p[j][1] - p[j][0] * p[i][1]
        return abs(area) * 0.5

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Initialize with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Local search with perturbations
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Try perturbing each point
        for i in range(N):
            if time.time() >= deadline:
                break
            for _ in range(20):
                if time.time() >= deadline:
                    break
                trial_pts = best_pts.copy()
                # Random perturbation
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
                trial_pts[i] += np.array([dx, dy])
                
                trial_score = _score(trial_pts)
                if trial_score > best_score + 1e-10:
                    best_pts = trial_pts
                    best_score = trial_score
                    improved = True
        
        # Try larger moves
        for i in range(N):
            if time.time() >= deadline:
                break
            for _ in range(5):
                if time.time() >= deadline:
                    break
                trial_pts = best_pts.copy()
                dx = np.random.normal(0, step_size * 2)
                dy = np.random.normal(0, step_size * 2)
                trial_pts[i] += np.array([dx, dy])
                
                trial_score = _score(trial_pts)
                if trial_score > best_score + 1e-10:
                    best_pts = trial_pts
                    best_score = trial_score
                    improved = True
        
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
        else:
            no_improve_count = 0
            # Slightly increase step size on improvement
            step_size = min(step_size * 1.05, 0.2)
    
    return best_pts
