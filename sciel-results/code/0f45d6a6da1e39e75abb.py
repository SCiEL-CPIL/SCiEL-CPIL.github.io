"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (n, 2) array
    """
    # points[TRI_IDX[:, 0]] gives shape (num_triangles, 2)
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 1e-10 # Degenerate case

def score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    best_score = -1.0
    best_points = None
    
    # Initial guesses
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Random points in unit square
    rng = np.random.default_rng(42)
    rand_points = rng.random((N, 2)) * 2 - 1 # [-1, 1]
    
    initializations = [reg_points, rand_points]
    
    # Add a few more random initializations
    for _ in range(5):
        initializations.append(rng.random((N, 2)) * 2 - 1)

    def objective(x):
        """
        Objective function for minimization.
        We want to maximize min_triangle_area.
        So we minimize -min_triangle_area.
        x is a flattened array of points.
        """
        pts = x.reshape(-1, 2)
        # Penalty for points going too far out (to keep scale consistent)
        # Though scale is invariant, numerical stability is better with bounds.
        # We use a soft bound or just rely on the optimizer staying near init.
        return -min_triangle_area(pts)
    
    for init in initializations:
        if time.time() > deadline - safety_margin:
            break
            
        x0 = init.flatten()
        
        # Use Nelder-Mead as it's robust for small dimensions (26 vars)
        # and doesn't require gradients.
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < -1e-15:
                pts = res.x.reshape(-1, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

    # If optimization failed or didn't improve, fallback to best initial
    if best_points is None:
        best_points = initializations[0]
        best_score = score(best_points)

    # Final check and return
    # The problem asks for the array of points.
    # We should probably normalize the points to unit area hull?
    # "The notebook normalizes by the point set convex hull area."
    # This usually means the score calculation does it.
    # The return value is just the points.
    return best_points

# EVOLVE-BLOCK-END
