# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 40s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _objective(points_flat):
    """
    Objective function for minimization (we minimize negative score).
    """
    points = points_flat.reshape(13, 2)
    s = _score(points)
    # If score is 0, it's invalid (collinear or degenerate), penalize heavily
    if s <= 1e-12:
        return 1e6
    return -s

def _generate_initial_points(seed):
    """
    Generate diverse initial point configurations.
    """
    n = 13
    rng = np.random.default_rng(seed)
    strategy = seed % 3
    
    if strategy == 0:
        # Random uniform
        return rng.random((n, 2))
        
    elif strategy == 1:
        # Grid-like: 4x4 grid, take 13, add noise
        x = np.linspace(0, 1, 4)
        y = np.linspace(0, 1, 4)
        grid = np.array([[xi, yi] for xi in x for yi in y])
        idx = rng.permutation(16)
        points = grid[idx[:13]].copy()
        points += rng.normal(0, 0.05, (n, 2))
        return points
        
    else:
        # Hexagonal-like: Center + ring
        # 1 center, 6 in first ring, 6 in second ring? 
        # Let's just use a regular 13-gon + center? No, 13 points total.
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = np.ones(n) * 0.8
        # Perturb radii and angles
        radii += rng.normal(0, 0.1, n)
        angles += rng.normal(0, 0.1, n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    # Run multiple restarts
    current_seed = 0
    max_restarts = 200 # Cap to prevent infinite loop if time check fails
    
    while time.time() < _DEADLINE and current_seed < max_restarts:
        # Check deadline more frequently
        if time.time() > _DEADLINE - 10:
            break
            
        initial_points = _generate_initial_points(current_seed)
        initial_flat = initial_points.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _objective, 
                initial_flat, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
        except Exception:
            current_seed += 1
            continue
        
        res_points = res.x.reshape(n, 2)
        res_score = _score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points.copy()
            
        current_seed += 1
        
    return best_points

# EVOLVE-BLOCK-END
