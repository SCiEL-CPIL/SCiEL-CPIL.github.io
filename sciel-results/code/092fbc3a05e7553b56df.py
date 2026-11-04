import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    # Ensure points are 2D
    points = np.asarray(points)
    n = len(points)
    
    if n < 3:
        return 0.0
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0 # Degenerate hull
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    min_area = float('inf')
    # Vectorized: compute all C(n,3) triangles
    indices = np.arange(n)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-15:
                        return 0.0  # Early exit for degenerate
                
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def _optimize_from_start(start_points):
    """Run Nelder-Mead from a given starting point."""
    x0 = start_points.flatten()
    
    def objective(x):
        points = x.reshape(-1, 2)
        return _score(points)
        
    # Nelder-Mead with careful parameters
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7,
                            'adaptive': True})
    
    return res.x.reshape(-1, 2), res.fun

def construct():
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    base_points = np.column_stack((x, y))
    
    best_points = base_points
    best_score = _score(base_points)
    
    # Try multiple perturbations and optimizations
    rng = np.random.default_rng(42)
    
    # First, try the base
    opt_points, opt_score = _optimize_from_start(base_points)
    if -opt_score > best_score:
        best_points = opt_points
        best_score = -opt_score
    
    # Try several perturbed starts
    for trial in range(5):
        # Small random perturbation
        noise = rng.normal(0, 0.05, size=(N, 2))
        perturbed = base_points + noise
        # Re-center
        perturbed -= perturbed.mean(axis=0)
        
        opt_points, opt_score = _optimize_from_start(perturbed)
        if -opt_score > best_score:
            best_points = opt_points
            best_score = -opt_score
    
    return best_points
# EVOLVE-BLOCK-END
