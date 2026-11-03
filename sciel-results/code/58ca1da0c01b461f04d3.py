"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _all_triangle_areas(points):
    """Compute all triangle areas from combinations of 3 points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, k + 1) if False else range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _is_inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Check using barycentric coordinates or half-plane tests
    # Edge 1: from (0,0) to (1,0) -> y >= 0
    if y < -tol:
        return False
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2) -> line: y = -sqrt(3)(x - 1)
    # Point should be below or on this line: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + tol:
        return False
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0) -> line: y = sqrt(3)*x
    # Point should be below or on this line: y <= sqrt(3)*x
    if y > np.sqrt(3) * x + tol:
        return False
    return True

def _project_to_triangle(p):
    """Project a point to the nearest point inside the triangle (approximate)."""
    x, y = p
    # Simple projection: clip to the triangle
    # For simplicity, if outside, move toward centroid
    cx, cy = 0.5, np.sqrt(3) / 6
    if not _is_inside_triangle(p):
        # Move toward centroid
        t = 0.5
        x = x + t * (cx - x)
        y = y + t * (cy - y)
        # May still be outside, so iterate
        for _ in range(10):
            if _is_inside_triangle([x, y]):
                break
            t = 0.5
            x = x + t * (cx - x)
            y = y + t * (cy - y)
    return [x, y]

def _min_area(points):
    """Return the minimum triangle area among all triplets."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _objective(x):
    """Objective function: negative of minimum area (we minimize this)."""
    points = x.reshape(N, 2)
    # Project points to triangle
    proj_points = []
    for i in range(N):
        p = _project_to_triangle(points[i])
        proj_points.append(p)
    proj_points = np.array(proj_points)
    ma = _min_area(proj_points)
    return -ma

def _initial_guess_grid():
    """Generate an initial guess using a structured grid inside the triangle."""
    # Use a hexagonal or grid pattern
    points = []
    # Place points on a grid, keeping only those inside the triangle
    # Scale to fit within the triangle
    # The triangle has width 1 and height sqrt(3)/2
    # Use a coarse grid
    for i in range(5):
        for j in range(5):
            # Map to triangle coordinates
            # Use barycentric-like mapping
            u = i / 4.0
            v = j / 4.0
            # Point in triangle: (u + v/2, v * sqrt(3)/2)
            x = u + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            if _is_inside_triangle([x, y]):
                points.append([x, y])
    # Shuffle and take first N
    np.random.seed(42)
    np.random.shuffle(points)
    points = points[:N]
    if len(points) < N:
        # Fill with random points
        while len(points) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (np.sqrt(3) / 2) * v])
    return np.array(points)

def _initial_guess_random():
    """Generate a random initial guess."""
    np.random.seed(123)
    points = []
    while len(points) < N:
        u, v = np.random.random(), np.random.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, (np.sqrt(3) / 2) * v])
    return np.array(points)

def _optimize_from_start(start_points):
    """Run Nelder-Mead optimization from a starting point."""
    x0 = start_points.flatten()
    # Nelder-Mead parameters
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    return result.x.reshape(N, 2), result.fun

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    best_points = None
    best_score = -float('inf')
    
    # Try multiple initial guesses
    starts = [_initial_guess_grid(), _initial_guess_random()]
    
    # Add a few more random starts
    np.random.seed(7)
    for seed in range(5):
        np.random.seed(seed * 100 + 1)
        pts = []
        while len(pts) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, (np.sqrt(3) / 2) * v])
        starts.append(np.array(pts))
    
    for start in starts:
        try:
            opt_points, score = _optimize_from_start(start)
            # Project to ensure inside triangle
            proj_points = []
            for i in range(N):
                p = _project_to_triangle(opt_points[i])
                proj_points.append(p)
            proj_points = np.array(proj_points)
            # Compute actual min area
            ma = _min_area(proj_points)
            if ma > best_score:
                best_score = ma
                best_points = proj_points
        except Exception as e:
            continue
    
    if best_points is None:
        # Fallback to random
        best_points = _initial_guess_random()
    
    # Ensure all points are strictly inside
    final_points = []
    for i in range(N):
        p = _project_to_triangle(best_points[i])
        # Add small inward shift to ensure strictly inside
        cx, cy = 0.5, np.sqrt(3) / 6
        p = [0.99 * p[0] + 0.01 * cx, 0.99 * p[1] + 0.01 * cy]
        if not _is_inside_triangle(p):
            p = [cx, cy]
        final_points.append(p)
    
    return np.array(final_points)
# EVOLVE-BLOCK-END
