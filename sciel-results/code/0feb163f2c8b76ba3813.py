import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    
    def compute_ratio(points):
        """
        Compute A_min / A_hull.
        """
        # Calculate areas of all triangles
        # points shape: (n, 2)
        # tri_indices shape: (286, 3)
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Vectorized:
        # x1 = p1[:, 0], y1 = p1[:, 1], etc.
        areas = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        
        a_min = np.min(areas)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume  # In 2D, volume is area
        except Exception:
            # If hull fails (e.g., collinear), return 0
            return 0.0
            
        if a_hull == 0:
            return 0.0
            
        return a_min / a_hull

    def objective(x):
        """
        Objective function for minimization (negative of ratio).
        x is a flat array of shape (26,).
        """
        points = x.reshape((n, 2))
        # Add a small penalty if points are too close or collinear?
        # The ratio itself goes to 0 if collinear.
        ratio = compute_ratio(points)
        return -ratio

    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Jittered Grid
    # Create a 4x4 grid, remove 3 points, jitter
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    # Remove 3 points (e.g., indices 0, 1, 2)
    # Let's try removing corners to make it more symmetric?
    # Or just random removal.
    
    # Strategy 2: Random points in unit square
    # Strategy 3: Points on a circle (regular polygon)
    
    initializations = []
    
    # 1. Jittered 4x4 grid minus 3 points
    for _ in range(10):
        grid_copy = grid.copy()
        # Remove 3 random points
        remove_idx = rng.choice(16, 3, replace=False)
        grid_copy = np.delete(grid_copy, remove_idx, axis=0)
        # Jitter
        jitter = rng.normal(0, 0.1, size=(13, 2))
        init_points = grid_copy + jitter
        # Normalize to unit square roughly
        init_points = (init_points - init_points.min(axis=0)) / (init_points.max(axis=0) - init_points.min(axis=0))
        initializations.append(init_points)
        
    # 2. Random points
    for _ in range(20):
        init_points = rng.random((13, 2))
        initializations.append(init_points)
        
    # 3. Regular polygon (13 points on a circle)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(init_points)

    # Optimization loop
    for i, init_points in enumerate(initializations):
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(
                objective, 
                init_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            opt_points = res.x.reshape((n, 2))
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_points = opt_points
                
        except Exception as e:
            continue

    if best_points is None:
        # Fallback: return random points
        return rng.random((13, 2))
        
    return best_points

# EVOLVE-BLOCK-END
