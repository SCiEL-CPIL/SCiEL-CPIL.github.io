import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def compute_min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    n = points.shape[0]
    min_area = float('inf')
    # Vectorized calculation of all triangle areas
    # i < j < k
    # We can loop i, j and vectorize k, but N is small, so simple loops or pre-computed indices are fine.
    # Pre-compute indices for speed
    idx = np.array([i for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    idx2 = np.array([j for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    idx3 = np.array([k for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    
    p1 = points[idx]
    p2 = points[idx2]
    p3 = points[idx3]
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def compute_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or nearly so, hull area is 0 or invalid
        return 1e-10

def objective(x):
    """
    Objective function to minimize.
    We want to maximize min_triangle_area / hull_area.
    So we minimize - (min_triangle_area / hull_area).
    """
    points = x.reshape(-1, 2)
    min_area = compute_min_triangle_area(points)
    hull_area = compute_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e10  # Large penalty for degenerate hulls
    
    ratio = min_area / hull_area
    return -ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1e10
    best_points = None
    
    # Initial guesses
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0_regular = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # 2. Jittered grid
    # 4x4 grid is 16 points, we need 13.
    # Let's take a 4x4 grid and remove 3 points.
    grid = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [1.5, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0],
        [0.0, 1.5], [0.5, 1.5], [1.0, 1.5], [1.5, 1.5]
    ])
    # Center it
    grid -= np.mean(grid, axis=0)
    # Pick 13 points: remove 3 corners? Or just take first 13?
    # Better: random selection of 13 from 16, but deterministic for reproducibility in restarts?
    # Let's just use the first 13 for the initial guess.
    x0_grid = grid[:13].flatten()
    
    # 3. Random points in a circle
    rng = np.random.default_rng(42)
    angles_rand = rng.uniform(0, 2*np.pi, N)
    radii_rand = rng.uniform(0.5, 1.0, N)
    x0_random = np.column_stack([radii_rand * np.cos(angles_rand), radii_rand * np.sin(angles_rand)]).flatten()
    
    initial_guesses = [x0_regular, x0_grid, x0_random]
    
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success:
                points = res.x.reshape(-1, 2)
                min_area = compute_min_triangle_area(points)
                hull_area = compute_hull_area(points)
                if hull_area > 1e-10:
                    score = min_area / hull_area
                    if score > best_score:
                        best_score = score
                        best_points = points
        except Exception:
            pass

    # If optimization didn't find anything good, fallback to regular polygon
    if best_points is None:
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
