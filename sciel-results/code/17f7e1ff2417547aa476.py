import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _get_triangle_indices(n):
    """Precompute indices for all triangles."""
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices)

def _triangle_areas(points, indices):
    """
    Compute areas of all triangles defined by indices.
    points: (N, 2) array
    indices: (M, 3) array of (i, j, k)
    Returns: (M,) array of areas
    """
    p_i = points[indices[:, 0]]
    p_j = points[indices[:, 1]]
    p_k = points[indices[:, 2]]
    
    x_i, y_i = p_i[:, 0], p_i[:, 1]
    x_j, y_j = p_j[:, 0], p_j[:, 1]
    x_k, y_k = p_k[:, 0], p_k[:, 1]
    
    areas = 0.5 * np.abs(x_i * (y_j - y_k) + x_j * (y_k - y_i) + x_k * (y_i - y_j))
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def _exact_objective(params, n, indices):
    """
    Objective function to minimize: - (A_min / A_hull).
    Uses exact min.
    """
    points = params.reshape(-1, 2)
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    areas = _triangle_areas(points, indices)
    a_min = np.min(areas)
    
    a_hull = _hull_area(points)
    if a_hull < 1e-9:
        return 1e10
        
    ratio = a_min / a_hull
    return -ratio

def _generate_initializers(n, rng):
    """
    Generate a list of diverse initial point configurations.
    """
    initializers = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    initializers.append(regular)
    
    # 2. Concentric rings
    if n >= 13:
        # Config 1: 1 center, 5 ring, 7 ring
        p_center = np.array([[0.0, 0.0]])
        angles1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring1 = 0.5 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring2 = 1.0 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        initializers.append(np.vstack([p_center, p_ring1, p_ring2]))
        
        # Config 2: 2 ring, 5 ring, 6 ring
        angles1 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring1 = 0.3 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        angles2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring2 = 0.6 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        angles3 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring3 = 1.0 * np.column_stack([np.cos(angles3), np.sin(angles3)])
        initializers.append(np.vstack([p_ring1, p_ring2, p_ring3]))
        
        # Config 3: 3 ring, 4 ring, 6 ring
        angles1 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring1 = 0.4 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring2 = 0.7 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        angles3 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi)
        p_ring3 = 1.0 * np.column_stack([np.cos(angles3), np.sin(angles3)])
        initializers.append(np.vstack([p_ring1, p_ring2, p_ring3]))

    # 3. Grid with jitter
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    jittered_grid = grid_points[:n] + rng.normal(0, 0.1, (n, 2))
    initializers.append(jittered_grid)

    # 4. Random points in unit disk
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0.2, 1.0, n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializers.append(pts)

    # 5. Random points in square
    for _ in range(5):
        pts = rng.uniform(-1, 1, (n, 2))
        initializers.append(pts)

    return initializers

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    Uses Nelder-Mead optimization on the exact objective with diverse initializers.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin for 600s limit
    
    n = 13
    indices = _get_triangle_indices(n)
    
    best_ratio = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    initializers = _generate_initializers(n, rng)
    
    # Shuffle initializers
    rng.shuffle(initializers)
    
    for init_points in initializers:
        if time.time() > deadline - 10.0:
            break
            
        x0 = init_points.flatten()
        
        try:
            result = minimize(
                _exact_objective, 
                x0, 
                args=(n, indices),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            points = result.x.reshape(-1, 2)
            
            areas = _triangle_areas(points, indices)
            a_min = np.min(areas)
            a_hull = _hull_area(points)
            
            if a_hull > 1e-9 and a_min > 0:
                ratio = a_min / a_hull
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
                    
        except Exception:
            continue

    # Final refinement
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, 
                x0, 
                args=(n, indices),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            points = result.x.reshape(-1, 2)
            areas = _triangle_areas(points, indices)
            a_min = np.min(areas)
            a_hull = _hull_area(points)
            if a_hull > 1e-9 and a_min > 0:
                ratio = a_min / a_hull
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
