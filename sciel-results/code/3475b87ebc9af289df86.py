# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0


def _min_triangle_area_and_indices(points):
    """Find the minimum triangle area and its indices."""
    n = len(points)
    min_area = np.inf
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx


def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area


def _compute_forces(points, min_area, min_idx, hull_mask, repulse_strength=1.0, smooth_strength=0.01):
    """
    Compute forces on each point:
    - Repulsive force from the minimum-area triangle's vertices (push them apart)
    - Laplacian smoothing toward the average of neighbors (for stability)
    - Hull points get a gentle inward pull to maintain convexity
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Repulsive force: push vertices of the smallest triangle apart
    i, j, k = min_idx
    # Force on each vertex to move away from the centroid of the triangle
    centroid = (points[i] + points[j] + points[k]) / 3.0
    
    for idx in [i, j, k]:
        to_centroid = centroid - points[idx]
        # Push away from centroid (increase area)
        forces[idx] -= to_centroid * repulse_strength * (min_area / 0.01)  # scaled
    
    # Laplacian smoothing: each point moves toward average of its Delaunay neighbors
    try:
        tri = Delaunay(points)
        for idx in range(n):
            neighbors = []
            for simplex in tri.simplices:
                if idx in simplex:
                    for other in simplex:
                        if other != idx:
                            neighbors.append(other)
            if neighbors:
                neighbor_avg = points[neighbors].mean(axis=0)
                forces[idx] += (neighbor_avg - points[idx]) * smooth_strength
    except Exception:
        pass
    
    # Hull points: gentle inward pull to prevent degeneration
    if hull_mask.any():
        hull_centroid = points[hull_mask].mean(axis=0)
        for idx in np.where(hull_mask)[0]:
            to_centroid = hull_centroid - points[idx]
            forces[idx] += to_centroid * 0.001
    
    return forces


def _initial_hexagonal_packing(n=13):
    """
    Create an initial configuration: hexagonal packing with some points on a convex hull.
    We place points in a hexagonal grid pattern and add hull points.
    """
    points = []
    
    # Central point
    points.append([0.0, 0.0])
    
    # First ring (6 points) at distance 1
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    
    # Second ring: add 6 more points at distance 2, but we only need 6 more to reach 13
    # Place them at angles offset by 30 degrees
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        points.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    
    return np.array(points[:n])


def _random_initial(n=13, seed=42):
    """Generate a random initial configuration inside a convex shape."""
    rng = np.random.default_rng(seed)
    # Generate points in a disk, then add some hull points
    points = []
    # 5 hull points forming a convex polygon
    angles = np.sort(rng.uniform(0, 2*np.pi, 5))
    for a in angles:
        r = rng.uniform(0.8, 1.0)
        points.append([r * np.cos(a), r * np.sin(a)])
    # 8 interior points
    for _ in range(8):
        r = rng.uniform(0, 0.7)
        a = rng.uniform(0, 2*np.pi)
        points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:n])


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of minimum triangle area
    to convex hull area.
    
    Uses physics-based relaxation with:
    - Initial hexagonal packing configuration
    - Force-based optimization pushing apart vertices of smallest triangles
    - Laplacian smoothing for stability
    - Multiple restarts with different initial configurations
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, keep safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Hexagonal packing
    initial_configs.append(_initial_hexagonal_packing(n))
    
    # 2. Random configurations with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        initial_configs.append(_random_initial(n, seed=seed))
    
    # 3. Regular polygon with center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)])
    poly_points = np.vstack([poly_points, [[0.0, 0.0]]])
    initial_configs.append(poly_points)
    
    for init_idx, points in enumerate(initial_configs):
        if time.time() > deadline - 60:  # 60s safety margin
            break
        
        # Normalize to unit area hull
        hull_area = _hull_area(points)
        if hull_area > 0:
            scale = 1.0 / np.sqrt(hull_area)
            points = points * scale
        
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Relaxation iterations
        max_iters = 5000
        learning_rate = 0.01
        min_improvement = 1e-8
        
        for iteration in range(max_iters):
            if time.time() > deadline - 60:
                break
            
            min_area, min_idx = _min_triangle_area_and_indices(points)
            hull_area = _hull_area(points)
            if hull_area <= 0:
                break
            
            current_score = min_area / hull_area
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Determine hull mask
            try:
                hull = ConvexHull(points)
                hull_mask = np.zeros(n, dtype=bool)
                for simplex in hull.simplices:
                    for idx in simplex:
                        hull_mask[idx] = True
            except Exception:
                hull_mask = np.zeros(n, dtype=bool)
            
            # Compute forces
            forces = _compute_forces(
                points, min_area, min_idx, hull_mask,
                repulse_strength=1.0, smooth_strength=0.005
            )
            
            # Update positions
            points = points + learning_rate * forces
            
            # Ensure no NaN/inf
            if not np.all(np.isfinite(points)):
                break
            
            # Adaptive learning rate
            if iteration % 100 == 0 and iteration > 0:
                learning_rate *= 0.95
        
        # Final evaluation
        final_score = _score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
    
    # If no valid result, fall back to a simple configuration
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _initial_hexagonal_packing(n)
    
    return best_points


# EVOLVE-BLOCK-END
