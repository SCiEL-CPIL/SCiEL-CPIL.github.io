"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _compute_min_triangle_area_vectorized(points):
    """
    Compute the minimum area of all triangles formed by pairs of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # points shape: (n, 2)
    # We need to compute area for all triples (i, j, k) with i < j < k.
    # Instead of iterating, we can compute the cross product for all pairs and then find the min?
    # No, area depends on 3 points.
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # Let's use broadcasting to compute areas for all i, j, k where i<j<k is not strictly necessary for min,
    # we can just compute all distinct triples.
    # Number of triples is C(14,3) = 364. This is small enough to loop in Python if needed, 
    # but vectorizing the inner loop or using full broadcasting is faster.
    
    # Full broadcasting approach:
    # x1 = points[:, 0][:, None, None] # (n, 1, 1)
    # y1 = points[:, 1][:, None, None]
    # x2 = points[:, 0][None, :, None] # (1, n, 1)
    # y2 = points[:, 1][None, :, None]
    # x3 = points[:, 0][None, None, :] # (1, 1, n)
    # y3 = points[:, 1][None, None, :]
    
    # This creates (n, n, n) arrays which is 14^3 = 2744 elements. Very fast.
    x1 = points[:, 0][:, None, None]
    y1 = points[:, 1][:, None, None]
    x2 = points[:, 0][None, :, None]
    y2 = points[:, 1][None, :, None]
    x3 = points[:, 0][None, None, :]
    y3 = points[:, 1][None, None, :]
    
    # Cross product z-component
    cross = (x2 - x1) * (y3 - y1) - (x2 - x1) * 0 # Wait, formula is (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    # We only want i < j < k.
    # Create a mask for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract areas where mask is true
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
        
    return np.min(valid_areas)

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9 # Avoid division by zero

def _score(points):
    """
    Score: minimum triangle area divided by convex hull area.
    """
    min_area = _compute_min_triangle_area_vectorized(points)
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    return min_area / hull_area

def construct():
    """
    Use Simulated Annealing to maximize the score.
    """
    np.random.seed(42)
    
    # Initial points: Regular polygon inscribed in unit circle (scaled)
    # Or random points. Regular polygon is a good symmetric start.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 0.5 + 0.5 * np.cos(angles)
    y = 0.5 + 0.5 * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Ensure points are within [0,1]x[0,1] roughly, but hull area normalization handles scaling.
    # Actually, we can let them be anywhere, the ratio is scale-invariant.
    # But keeping them in a reasonable box helps numerical stability.
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.01  # Initial temperature
    alpha = 0.999 # Cooling factor
    T_min = 1e-6
    max_iter = 50000
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor: perturb one point
        idx = np.random.randint(N)
        # Gaussian perturbation
        step = 0.05 * np.random.randn(2)
        new_points = points.copy()
        new_points[idx] += step
        
        # Optional: keep points within a reasonable bound to avoid explosion
        # Though score is scale invariant, huge coordinates might cause floating point issues.
        # Let's clip to [-10, 10] just in case.
        new_points = np.clip(new_points, -10, 10)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    points = new_points
                    current_score = new_score
        
        T *= alpha
        if T < T_min:
            break
            
    return best_points
# EVOLVE-BLOCK-END
