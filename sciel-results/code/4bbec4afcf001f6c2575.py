"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    h = np.sqrt(3) / 2
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, h])
    
    v0 = B - A
    v1 = C - A
    v2 = p - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    
    return u >= -tol and v >= -tol and (u + v) <= 1 + tol

def _is_valid(points):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p) for p in points)

def _get_critical_triangles(points, n_critical=5):
    """
    Get the indices of the n_critical smallest triangles.
    Returns a list of (i, j, k) tuples.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                areas.append((area, (i, j, k)))
    
    areas.sort(key=lambda x: x[0])
    return [idx for _, idx in areas[:n_critical]]

def _objective_focused(points_flat, critical_indices):
    """
    Objective function that focuses on the smallest triangles.
    Returns negative of the minimum area among critical triangles.
    """
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        return 1e10
    
    min_area = float('inf')
    for idx in critical_indices:
        i, j, k = idx
        area = _triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return -min_area

def _local_optimize(points, maxiter=300):
    """
    Local optimization using L-BFGS-B with focus on critical triangles.
    """
    h = np.sqrt(3) / 2
    points = np.array(points)
    
    # Get critical triangles
    critical = _get_critical_triangles(points, n_critical=5)
    
    def obj(x):
        return _objective_focused(x, critical)
    
    # Bounds for each coordinate
    bounds = [(0, 1), (0, h)] * N
    
    res = minimize(
        obj,
        points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if res.success and _is_valid(res.x.reshape(-1, 2)):
        return res.x.reshape(-1, 2)
    else:
        return points

def _initial_guesses():
    """Generate diverse initial guesses."""
    h = np.sqrt(3) / 2
    guesses = []
    
    # Random with different seeds
    import random
    for seed in [42, 7, 123, 456, 789]:
        rng = random.Random(seed)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.05 and v > 0.05:
                pts.append([u + 0.5 * v, h * v])
        if _is_valid(pts):
            guesses.append(np.array(pts))
    
    # Grid-based patterns
    for scale in [2, 3]:
        pts = []
        for i in range(scale + 1):
            for j in range(scale + 1):
                if i + j <= scale:
                    u = i / scale
                    v = j / scale
                    pts.append([u + 0.5 * v, h * v])
        if len(pts) >= N:
            pts = pts[:N]
            if _is_valid(pts):
                guesses.append(np.array(pts))
    
    # Symmetric pattern: vertices, edge midpoints, center
    center = [0.5, h/3]
    edge_mids = [[0.5, 0], [0.25, h/2], [0.75, h/2]]
    vertices = [[0, 0], [1, 0], [0.5, h]]
    pts = vertices + edge_mids + [center]
    rng = random.Random(99)
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.9 and u > 0.1 and v > 0.1:
            p = [u + 0.5 * v, h * v]
            if _inside_triangle(p):
                pts.append(p)
    if _is_valid(pts):
        guesses.append(np.array(pts))
    
    return guesses

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    
    # Generate initial guesses
    guesses = _initial_guesses()
    
    best_points = None
    best_area = -1
    
    for guess in guesses:
        # Local optimization
        optimized = _local_optimize(guess, maxiter=300)
        area = _min_triangle_area(optimized)
        
        if area > best_area:
            best_area = area
            best_points = optimized
    
    # Final validation
    if best_points is None or not _is_valid(best_points):
        # Fallback
        import random
        rng = random.Random(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, h * v])
        best_points = np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
