# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
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
    
    # Calculate areas using cross product
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    Returns 0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
        
    return a_min / a_hull

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _distances_squared(points):
    """Compute squared distances between all pairs of points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    return dist2

def _hill_climb(points, max_iter=2000, initial_step=0.05, min_step=1e-4):
    """
    Perform hill climbing to maximize the score.
    Uses adaptive step sizes and coordinate-wise perturbations.
    """
    n = len(points)
    current_points = points.copy()
    current_score = _score(current_points)
    step = initial_step
    best_points = current_points.copy()
    best_score = current_score
    
    # Precompute pair indices for speed
    pair_indices = list(itertools.combinations(range(n), 2))
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        
        # Try to improve by moving each point in random direction
        for _ in range(n * 2):  # Try multiple perturbations per iteration
            idx = np.random.randint(n)
            direction = np.random.normal(size=2)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                
            new_points = current_points.copy()
            new_points[idx] += step * direction
            
            new_score = _score(new_points)
            
            if new_score > current_score + 1e-10:
                current_points = new_points
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    
        if not improved:
            step *= 0.8
            if step < min_step:
                break
        else:
            step *= 1.05
            if step > initial_step:
                step = initial_step
        
    return best_points, best_score

def _local_optimize(points, max_iterations=5000, initial_radius=0.02):
    """
    Local optimization: iteratively find the most constrained triangle and 
    move one of its points to reduce the constraint.
    """
    n = len(points)
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    tri_indices = list(itertools.combinations(range(n), 3))
    
    for it in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
            
        # Find the minimum area triangle
        p1 = current_points[tri_indices[:, 0]]
        p2 = current_points[tri_indices[:, 1]]
        p3 = current_points[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        min_tri = tri_indices[min_idx]
        
        # For the minimum triangle, try moving each of its 3 points
        # in the direction that increases the area of this triangle
        # while not decreasing other triangle areas too much
        
        best_move = None
        best_new_score = current_score
        
        for point_idx in min_tri:
            p = current_points[point_idx]
            
            # Try several directions
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                for magnitude in [0.01, 0.005, 0.002, 0.001]:
                    dx = magnitude * np.cos(angle)
                    dy = magnitude * np.sin(angle)
                    
                    new_points = current_points.copy()
                    new_points[point_idx] = p + np.array([dx, dy])
                    
                    new_score = _score(new_points)
                    
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_move = new_points.copy()
        
        if best_move is not None:
            current_points = best_move
            current_score = best_new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            break
            
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Start with regular polygon
    reg_poly = _regular_polygon(n, radius=1.0)
    s = _score(reg_poly)
    if s > best_score:
        best_score = s
        best_points = reg_poly.copy()
    
    # Strategy 2: Local optimization from regular polygon
    if time.time() < _DEADLINE - 30:
        opt_pts, opt_score = _local_optimize(reg_poly, max_iterations=2000, initial_radius=0.02)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Strategy 3: Hill climbing from various initializations
    rng = np.random.default_rng(seed=123)
    
    initializations = [best_points.copy()]
    
    # Add some random initializations
    for i in range(3):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.random(n) * 0.2
        radii = 1.0 + rng.random(n) * 0.2 - 0.1
        circ_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(circ_pts)
    
    # Hill climb from each initialization
    for init_pts in initializations:
        if time.time() > _DEADLINE - 20:
            break
        
        hc_pts, hc_score = _hill_climb(init_pts, max_iter=3000, initial_step=0.03, min_step=1e-5)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts.copy()
    
    # Final local optimization on the best points found
    if best_points is not None and time.time() < _DEADLINE - 10:
        final_pts, final_score = _local_optimize(best_points, max_iterations=3000, initial_radius=0.01)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    if best_points is None:
        best_points = reg_poly
        
    return best_points

# EVOLVE-BLOCK-END
