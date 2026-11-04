import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns the raw score (higher is better).
    """
    points = np.asarray(points)
    n = len(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    # We can vectorize this for speed.
    # Generate all combinations of 3 indices
    # C(13, 3) = 286, which is small enough to just loop or use itertools.
    # Let's use a fast loop or vectorized approach.
    
    min_area = float('inf')
    
    # Efficient way: iterate over triples
    # Since n=13 is small, a simple loop is fine, but let's make it robust.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (p2-p1) x (p3-p1) |
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    if min_area == 0:
        return 0.0
        
    return min_area / hull_area

def _negative_score(points):
    """
    Objective function for minimization.
    Returns negative score to maximize score.
    """
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    # 1. Regular polygon on circle
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    p_circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # 2. Grid-like distribution
    # 4x4 grid minus one corner? 13 points.
    # 3x5 grid = 15, remove 2.
    # Let's try a few random starts and the circle.
    
    initial_candidates = [p_circle]
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        initial_candidates.append(rng.rand(N, 2))
    
    # Optimize each candidate
    for i, x0 in enumerate(initial_candidates):
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(_negative_score, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.success or result.fun < -best_score:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to circle
        best_points = p_circle
        best_score = _score(p_circle)
        
    # Refinement step: Local search / Nelder-Mead on best found
    if time.time() < deadline - 20 and best_points is not None:
        try:
            # Nelder-Mead can be good for final polishing
            result_nm = minimize(_negative_score, best_points, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12})
            if -result_nm.fun > best_score:
                best_score = -result_nm.fun
                best_points = result_nm.x
        except Exception:
            pass
            
    # Final check
    if best_points is None:
        return np.array([[rng.random(), rng.random()] for _ in range(N)])
        
    return best_points
# EVOLVE-BLOCK-END
