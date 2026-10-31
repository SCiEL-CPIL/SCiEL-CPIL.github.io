"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools
import time

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
    # Using barycentric coordinates
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.5, h])
    v2 = np.array([p[0], p[1]])
    
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

def _objective(points_flat):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = points_flat.reshape(-1, 2)
    if not _is_valid(points):
        return 1e10  # Large penalty for invalid points
    
    min_area = _min_triangle_area(points)
    return -min_area

def _generate_initial_points(seed=42):
    """Generate initial points using a structured approach."""
    h = np.sqrt(3) / 2
    rng = np.random.RandomState(seed)
    
    # Use a hexagonal-like pattern for better initial distribution
    # Place points in a triangular lattice inside the triangle
    points = []
    
    # Try multiple configurations and pick the best
    best_points = None
    best_area = -1
    
    # Configuration 1: Triangular lattice
    for scale in [3, 4, 5]:
        pts = []
        for i in range(scale + 1):
            for j in range(scale + 1 - i):
                # Barycentric coordinates
                u = (i + 0.5) / (scale + 1)
                v = (j + 0.5) / (scale + 1)
                if u + v < 1.0:
                    # Convert to Cartesian
                    x = u + 0.5 * v
                    y = h * v
                    pts.append([x, y])
        
        # Add more points if needed
        while len(pts) < N:
            # Add random points near existing ones
            idx = rng.randint(0, len(pts))
            base = pts[idx]
            x = base[0] + rng.uniform(-0.05, 0.05)
            y = base[1] + rng.uniform(-0.05, 0.05)
            if _inside_triangle([x, y]):
                pts.append([x, y])
        
        pts = pts[:N]
        if _is_valid(pts):
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts
    
    # Configuration 2: Random with rejection
    for trial in range(50):
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                # Check minimum distance from existing points
                min_dist = min(np.sqrt((x - p[0])**2 + (y - p[1])**2) for p in pts) if pts else 1.0
                if min_dist > 0.05:
                    pts.append([x, y])
        
        if len(pts) >= N:
            pts = pts[:N]
            if _is_valid(pts):
                area = _min_triangle_area(pts)
                if area > best_area:
                    best_area = area
                    best_points = pts
    
    # Configuration 3: Known good starting point (vertices + centroid + others)
    vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, h]]
    centroid = [0.5, h/3]
    pts = vertices + [centroid]
    
    # Add 7 more points
    for i in range(7):
        angle = 2 * np.pi * i / 7
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = h/3 + r * np.sin(angle) * 0.5  # Scale y for triangle shape
        if _inside_triangle([x, y]):
            pts.append([x, y])
        else:
            # Fallback
            u = rng.random() * 0.8
            v = rng.random() * 0.8
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                pts.append([x, y])
    
    if len(pts) >= N:
        pts = pts[:N]
        if _is_valid(pts):
            area = _min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts
    
    if best_points is None:
        # Fallback to simple random
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                pts.append([x, y])
        best_points = pts
    
    return np.array(best_points).flatten()

def _refine_with_local_optimization(points):
    """Refine points using local optimization with multiple methods."""
    h = np.sqrt(3) / 2
    best_points = points
    best_score = _min_triangle_area(points)
    
    # Try multiple local optimization methods
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    
    for method in methods:
        try:
            res = minimize(
                _objective,
                best_points.flatten(),
                method=method,
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success and _is_valid(res.x.reshape(-1, 2)):
                new_score = _min_triangle_area(res.x.reshape(-1, 2))
                if new_score > best_score:
                    best_score = new_score
                    best_points = res.x.reshape(-1, 2)
        except:
            pass
    
    return best_points

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the smallest triangle area."""
    h = np.sqrt(3) / 2
    start_time = time.time()
    time_limit = 110  # Leave 10 seconds buffer
    
    # Get initial guess
    x0 = _generate_initial_points()
    
    # Define bounds for each coordinate
    bounds = [(0, 1), (0, h)] * N
    
    # Use differential evolution for global optimization with time management
    result = differential_evolution(
        _objective,
        bounds=bounds,
        seed=42,
        maxiter=50,
        tol=1e-6,
        polish=False,
        init='sobol',
        updating='deferred'
    )
    
    best_points = result.x.reshape(-1, 2)
    
    # Check time
    if time.time() - start_time > time_limit * 0.7:
        # Skip refinement if running out of time
        if _is_valid(best_points):
            return best_points
        else:
            # Fallback
            x0 = _generate_initial_points()
            return x0.reshape(-1, 2)
    
    # Refine with local optimization
    if _is_valid(best_points):
        best_points = _refine_with_local_optimization(best_points)
    
    # Final validation
    if not _is_valid(best_points):
        # Fallback to a valid configuration
        x0 = _generate_initial_points()
        best_points = x0.reshape(-1, 2)
    
    return best_points
# EVOLVE-BLOCK-END
