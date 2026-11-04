"""Heilbronn convex regions n=14 - optimized point placement."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_start_time = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
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
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_triangle_area(points, beta=50.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas[i, j, k] = area
                areas[k, j, i] = area
    # Use log-sum-exp for soft minimum
    log_sum = np.logsumexp(-beta * areas)
    soft_min = -np.log(np.exp(log_sum) / (n*(n-1)*(n-2)/6.0)) / beta
    return soft_min

def _init_rings(n_outer=8, n_inner=6):
    """Initialize points in concentric rings."""
    points = []
    # Outer ring on unit circle
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        points.append([np.cos(angle), np.sin(angle)])
    # Inner ring at smaller radius
    r_inner = 0.5
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + np.pi / n_inner
        points.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    return np.array(points[:N])

def _optimize_local(points, max_iter=200):
    """Local optimization to maximize min triangle area / hull area."""
    def objective(x):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area <= 1e-10:
            return 0.0
        min_area = _min_triangle_area(pts)
        return -min_area / hull_area
    
    # Nelder-Mead optimization
    res = minimize(objective, points.flatten(), method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    return res.x.reshape(-1, 2)

def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Return a reasonable default if we're out of time
        return _init_rings(8, 6)
    
    best_points = _init_rings(8, 6)
    best_score = _score(best_points)
    
    # Try multiple initializations and local optimizations
    initializations = [
        lambda: _init_rings(8, 6),
        lambda: _init_rings(7, 7),
        lambda: _init_rings(6, 8),
        lambda: _init_rings(9, 5),
        lambda: _init_rings(10, 4),
    ]
    
    # Add some perturbed versions
    for _ in range(3):
        def make_perturbed(base):
            def perturbed():
                pts = base()
                noise = np.random.randn(N, 2) * 0.1
                return pts + noise
            return perturbed
        initializations.append(make_perturbed(_init_rings(8, 6)))
    
    for init_func in initializations:
        elapsed = time.time() - _start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        try:
            points = init_func()
            # Local optimization
            optimized = _optimize_local(points, max_iter=100)
            score = _score(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    elapsed = time.time() - _start_time
    if elapsed < TIME_LIMIT - 30:
        try:
            polished = _optimize_local(best_points, max_iter=300)
            score = _score(polished)
            if score > best_score:
                best_points = polished
                best_score = score
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
