# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import basinhopping
import time

def _calculate_score(points):
    """
    Calculate the Heilbronn score: A_min / A_hull.
    """
    n = len(points)
    # Check for degeneracy
    if n < 3:
        return 0.0
    
    # Calculate areas of all triangles
    # Indices for i < j < k
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                 p2[0] * (p3[1] - p1[1]) + 
                                 p3[0] * (p1[1] - p2[1]))
                areas.append(area)
    
    a_min = min(areas)
    
    if a_min < 1e-10:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
        
    if a_hull < 1e-10:
        return 0.0

    return a_min / a_hull

def _neg_score(points_flat):
    """
    Objective function for minimization.
    Takes a flat array of 26 coordinates (13 points * 2).
    """
    points = points_flat.reshape(13, 2)
    score = _calculate_score(points)
    return -score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn score.
    Uses basin hopping with L-BFGS-B local optimization.
    """
    n = 13
    best_points = None
    best_score = -1.0
    
    # Time limit check
    start_time = time.time()
    time_limit = 580  # Leave some buffer for the 600s limit
    
    # Initial guesses
    # 1. Grid-like initialization
    # Create a 4x4 grid, remove 3 points
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Remove 3 points from the 16 grid points to get 13
    # Remove corners to keep the center dense? Or remove to make it more compact?
    # Let's just take the first 13 points of the grid (row-major)
    init_grid = grid_points[:13]
    
    # 2. Random initialization
    rng = np.random.default_rng(seed=42)
    init_random = rng.random((13, 2))
    
    # 3. Triangular lattice initialization (subset)
    # Generate a triangular lattice in a square
    lattice_points = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + i * 0.2 + (j % 2) * 0.1
            y = 0.1 + j * 0.2
            if 0 <= x <= 1 and 0 <= y <= 1:
                lattice_points.append([x, y])
    init_lattice = np.array(lattice_points[:13])
    
    initial_guesses = [init_grid, init_random, init_lattice]
    
    bounds = [(0, 1), (0, 1)] * 13
    
    for x0 in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        x0_flat = x0.ravel()
        
        try:
            res = basinhopping(
                _neg_score,
                x0_flat,
                niter=50,
                stepsize=0.5,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "bounds": bounds,
                    "maxiter": 1000
                }
            )
            
            current_points = res.x.reshape(13, 2)
            current_score = _calculate_score(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
        except Exception as e:
            pass

    # If basin hopping didn't find anything good, fall back to a simple grid
    if best_points is None:
        best_points = init_grid

    return best_points

# EVOLVE-BLOCK-END
