"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by three points.
    points: array of shape (N, 2)
    Returns: min_area
    """
    n = points.shape[0]
    min_area = np.inf
    
    # Vectorized calculation for speed
    # We can compute all pairs differences, but N=14 is small enough for loops
    # However, vectorizing the triple loop is tricky.
    # Let's stick to efficient loops or a semi-vectorized approach.
    
    # For N=14, number of triples is C(14,3) = 364.
    # This is very fast.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                # 2D cross product magnitude
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    # For N=14, we can use scipy.spatial.ConvexHull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback if points are collinear or degenerate (unlikely in optimization)
        # Just return a small epsilon to avoid division by zero
        return 1e-10

def _score(points):
    """
    Score is min_triangle_area / convex_hull_area
    """
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _negative_score(flat_points):
    """
    Objective function for minimizer.
    Reshapes flat array to (N, 2) and returns negative score.
    """
    points = flat_points.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: Regular grid or circle?
    # A regular grid often has good properties.
    # Or points on a circle.
    # Let's try a grid first, slightly perturbed to avoid symmetry locks if any, 
    # though Nelder-Mead should handle it.
    
    # Create a grid
    side = int(np.ceil(np.sqrt(N)))
    # We need exactly N points.
    # If N is not a perfect square, we take the first N points of a larger grid.
    # Or better, use a specific pattern.
    
    # Let's use a circle initialization as it maximizes hull area initially.
    # Actually, mixing is good. Let's start with a regular polygon.
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    # Scale to unit area? The score is invariant to scaling (area/area).
    # So we can work in any coordinate system.
    # However, numerical stability might prefer points near origin and unit scale.
    # Circle of radius 1 has area pi. Hull area is pi.
    
    # Let's try a few different initializations and pick the best start for optimization.
    candidates = []
    
    # 1. Regular polygon
    candidates.append(initial_points.copy())
    
    # 2. Grid
    # 4x4 grid = 16 points. Take first 14.
    # Grid points in [0,1]x[0,1]
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    if len(grid_points) >= N:
        candidates.append(grid_points[:N].copy())
    
    # 3. Random
    rng = np.random.RandomState(42)
    rand_points = rng.rand(N, 2)
    candidates.append(rand_points)

    best_score = -np.inf
    best_points = None
    
    for init_pts in candidates:
        if time.time() > deadline:
            break
            
        # Optimize from this start
        try:
            res = minimize(
                _negative_score, 
                init_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.success or res.fun < best_score:
                cur_score = -res.fun
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to random if optimization failed completely
        return np.array([[np.random.random(), np.random.random()] for _ in range(N)])
        
    return best_points

# EVOLVE-BLOCK-END
