"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _triangle_area_vec(p1, p2, p3):
    """Vectorized area of triangles with p1 as base, p2 and p3 varying."""
    # p1: (n,2), p2: (m,2), p3: (k,2)
    # area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2[None, :, :] - p1[:, None, :]  # (n, m, 2)
    v2 = p3[None, :, :] - p1[:, None, :]  # (n, k, 2)
    cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    return 0.5 * np.abs(cross)


def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(n,3) triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Use vectorized computation in chunks to avoid memory blowup
    min_area = float('inf')
    idx = np.arange(n)
    
    for i in range(n - 2):
        p1 = points[i:i+1]
        remaining = points[i+1:]
        m = len(remaining)
        if m < 2:
            continue
        
        # Compute all areas with p1 as one vertex
        # Remaining points: (m, 2)
        # We need all pairs from remaining
        v1 = remaining[:, None, :] - remaining[None, :, :]  # (m, m, 2)
        # For each pair (j, k) with j < k, area = 0.5 * |cross(p_j - p1, p_k - p1)|
        v1_from_p1 = remaining - p1[0]  # (m, 2)
        cross = v1_from_p1[:, 0] * v1_from_p1[None, :, 1] - v1_from_p1[:, 1] * v1_from_p1[None, :, 0]
        # Only use upper triangle
        cross = np.triu(cross, k=1)
        areas = 0.5 * np.abs(cross)
        if areas.size > 0:
            min_in_chunk = np.min(areas)
            if min_in_chunk < min_area:
                min_area = min_in_chunk
    
    return min_area


def _is_inside_triangle(points, tol=1e-9):
    """Check if all points are inside the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For point (x,y):
    #   y >= -tol
    #   y <= sqrt(3)*x + tol  (left edge from (0,0) to (0.5, sqrt(3)/2))
    #   y <= -sqrt(3)*(x-1) + tol  (right edge from (1,0) to (0.5, sqrt(3)/2))
    x = points[:, 0]
    y = points[:, 1]
    if np.any(y < -tol):
        return False
    if np.any(y > np.sqrt(3) * x + tol):
        return False
    if np.any(y > -np.sqrt(3) * (x - 1) + tol):
        return False
    return True


def _project_to_triangle(points):
    """Project points that are outside the triangle back inside."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip to y >= 0
    y = np.maximum(y, 0.0)
    
    # For each x, the max y is min(sqrt(3)*x, -sqrt(3)*(x-1)) for x in [0,1]
    # Left boundary: y <= sqrt(3)*x
    # Right boundary: y <= -sqrt(3)*(x-1)
    
    # First clip x to [0, 1]
    x = np.clip(x, 0.0, 1.0)
    
    # Max y for given x
    max_y_left = np.sqrt(3) * x
    max_y_right = -np.sqrt(3) * (x - 1)
    max_y = np.minimum(max_y_left, max_y_right)
    
    y = np.minimum(y, max_y)
    y = np.maximum(y, 0.0)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _objective(x_flat):
    """Objective to minimize: negative of min triangle area (we want to maximize)."""
    points = x_flat.reshape(N, 2)
    points = _project_to_triangle(points)
    min_a = _min_triangle_area(points)
    return -min_a


def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    # Initial random points inside triangle
    def random_point():
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        return [u + 0.5 * v, (np.sqrt(3) / 2) * v]
    
    np.random.seed(42)
    initial_points = np.array([random_point() for _ in range(N)])
    best_points = initial_points.copy()
    best_score = _min_triangle_area(best_points)
    
    # Use differential evolution for global optimization
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1)] * N + [(0, np.sqrt(3) / 2)] * N
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            polish=True,
            init='latinhypercube',
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.9,
            seed=42
        )
        
        if time.time() < deadline - 10:
            de_points = result.x.reshape(N, 2)
            de_points = _project_to_triangle(de_points)
            de_score = _min_triangle_area(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead on the best point
    if time.time() < deadline - 5:
        from scipy.optimize import minimize
        try:
            res = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            nm_points = res.x.reshape(N, 2)
            nm_points = _project_to_triangle(nm_points)
            nm_score = _min_triangle_area(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points
        except Exception:
            pass
    
    # Ensure points are inside triangle
    best_points = _project_to_triangle(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
