"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _refine(points):
    """Local refinement using Nelder-Mead."""
    def objective(x):
        return -_score(x.reshape(N, 2))
    
    # Use Nelder-Mead for non-smooth optimization
    res = minimize(
        objective,
        points.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    return res.x.reshape(N, 2)

def construct():
    start_time = time.time()
    best_score = 0.0
    best_points = None
    
    # Strategy: Try multiple initial configurations and refine each.
    # 1. Regular 14-gon
    # 2. Grid based
    # 3. Random
    
    initial_configs = []
    
    # Config 1: Regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(poly_points)
    
    # Config 2: Grid (4x4, remove 2 points to get 14)
    # Create a 4x4 grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    # Remove the center-ish points or corners? 
    # For max min area, we want spread. 
    # Let's just take the first 14 points of a shuffled grid? 
    # Better: Take a 3x5 grid? 15 points. Remove 1.
    grid_x2, grid_y2 = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 5))
    grid_points2 = np.column_stack([grid_x2.flatten(), grid_y2.flatten()])
    # Remove the last point
    initial_configs.append(grid_points2[:14])
    
    # Config 3: Random points in a circle
    np.random.seed(42)
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        rand_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_configs.append(rand_points)
        
    # Config 4: Square grid 4x4 minus 2 corners
    grid_x3, grid_y3 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_points3 = np.column_stack([grid_x3.flatten(), grid_y3.flatten()])
    # Remove top-right and bottom-left corners to keep convexity? 
    # Indices: 0,0 is (0,0) in meshgrid? No, meshgrid is (x, y).
    # Let's just use the 14 points from a 4x4 grid excluding the two opposite corners.
    # 4x4 = 16 points.
    # Points are indexed 0..15.
    # Corners: (0,0) -> 0, (3,0) -> 3, (0,3) -> 12, (3,3) -> 15.
    # Remove 0 and 15.
    mask = np.ones(16, dtype=bool)
    mask[0] = False
    mask[15] = False
    initial_configs.append(grid_points3[mask])

    for config in initial_configs:
        if time.time() - start_time > 100: # Leave 20s buffer
            break
        
        refined = _refine(config)
        score = _score(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined
            
    if best_points is None:
        # Fallback: return the regular polygon
        best_points = initial_configs[0]
        
    return best_points
# EVOLVE-BLOCK-END
