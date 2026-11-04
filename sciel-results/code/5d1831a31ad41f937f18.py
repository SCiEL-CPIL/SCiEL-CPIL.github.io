"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    # Vectorized calculation of all triangle areas
    # points shape: (N, 2)
    # We need areas of all triples (i, j, k) with i < j < k
    
    n = len(points)
    # Create indices
    i, j, k = np.indices((n, n, n))
    # Mask for i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Cross product (2D): (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    # Area = 0.5 * |cross|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Use Simulated Annealing starting from a regular polygon.
    """
    # Start with a regular 14-gon inscribed in the unit circle
    angles = 2 * np.pi * np.arange(N) / N
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_points)
    
    # Simulated Annealing parameters
    initial_temp = 0.5
    final_temp = 1e-4
    cooling_rate = 0.995
    n_iterations = 5000  # Adjust based on time limit
    
    rng = np.random.RandomState(42)
    current_points = best_points.copy()
    current_score = best_score
    temp = initial_temp
    
    for _ in range(n_iterations):
        # Generate a random perturbation
        # Move one point randomly
        idx = rng.randint(N)
        # Perturb size decreases with temperature
        step_size = temp * 0.5
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        new_points = current_points.copy()
        new_points[idx] += np.array([dx, dy])
        
        # Optional: Keep points within a bounding box to avoid degenerate hulls
        # Clip to [-1.5, 1.5] to allow some expansion but keep scale reasonable
        new_points = np.clip(new_points, -1.5, 1.5)
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / temp):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        temp *= cooling_rate
    
    return best_points
# EVOLVE-BLOCK-END
