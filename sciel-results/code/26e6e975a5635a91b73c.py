"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi
import warnings

warnings.filterwarnings('ignore')

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, eps=1e-10):
    """Check if point p is inside or on the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    # Check y >= 0
    if y < -eps:
        return False
    # Check y <= h
    if y > h + eps:
        return False
    # Check left edge: x >= y / sqrt(3)  (line from (0,0) to (0.5, h))
    if x < y / np.sqrt(3) - eps:
        return False
    # Check right edge: x <= 1 - y / sqrt(3)  (line from (1,0) to (0.5, h))
    if x > 1 - y / np.sqrt(3) + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the nearest point inside the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    
    # If already inside, return as is
    if _is_inside_triangle(p, eps=1e-8):
        return np.array([x, y])
    
    # Project to the three edges
    # Edge 1: y = 0, x in [0, 1]
    proj1 = np.array([np.clip(x, 0, 1), 0])
    
    # Edge 2: left edge from (0,0) to (0.5, h), line: x = y / sqrt(3)
    # Distance from (x,y) to line x - y/sqrt(3) = 0
    # Line in form ax + by + c = 0: x - y/sqrt(3) = 0
    a, b, c = 1, -1/np.sqrt(3), 0
    d = (a*x + b*y + c) / (a*a + b*b)
    proj2 = np.array([x - d*a, y - d*b])
    # Clamp to segment
    t = proj2[1] / h if h > 0 else 0
    t = np.clip(t, 0, 1)
    proj2 = np.array([t * 0.5, t * h])
    
    # Edge 3: right edge from (1,0) to (0.5, h), line: x = 1 - y/sqrt(3)
    a, b, c = 1, -1/np.sqrt(3), -1
    d = (a*x + b*y + c) / (a*a + b*b)
    proj3 = np.array([x - d*a, y - d*b])
    # Clamp to segment
    t = (1 - proj3[0]) / 0.5 if 0.5 > 0 else 0
    t = np.clip(t, 0, 1)
    proj3 = np.array([1 - t * 0.5, t * h])
    
    # Choose closest projection
    d1 = np.linalg.norm(p - proj1)
    d2 = np.linalg.norm(p - proj2)
    d3 = np.linalg.norm(p - proj3)
    
    min_d = min(d1, d2, d3)
    if min_d == d1:
        return proj1
    elif min_d == d2:
        return proj2
    else:
        return proj3

def _objective(params, points_init=None):
    """Objective function: negative of min triangle area (to minimize)."""
    n = N
    pts = params.reshape(n, 2)
    
    # Project points to triangle if they go outside
    for i in range(n):
        pts[i] = _project_to_triangle(pts[i])
    
    min_area = _min_triangle_area(pts)
    return -min_area

def _objective_with_penalty(params):
    """Objective with soft penalty for points too close to boundary or each other."""
    n = N
    pts = params.reshape(n, 2)
    
    # Project points to triangle
    for i in range(n):
        pts[i] = _project_to_triangle(pts[i])
    
    min_area = _min_triangle_area(pts)
    
    # Penalty for points being too close to each other (to avoid degeneracy)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 0.01:
                penalty += (0.01 - d) ** 2
    
    return -min_area + 0.1 * penalty

def _coordinate_descent(pts, max_iter=500, tol=1e-8, time_limit=None, start_time=None):
    """Coordinate descent refinement: move one point at a time to maximize min triangle area."""
    n = N
    best_val = -_min_triangle_area(pts)  # negative of min area
    pts = pts.copy()
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n):
            best_p = pts[i].copy()
            best_local_val = -_min_triangle_area(pts)
            
            # Try small perturbations in x and y
            for step in [0.01, 0.005, 0.001, 0.0005]:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (-step, step), (step, -step), (-step, -step)]:
                    new_p = pts[i] + np.array([dx, dy])
                    new_p = _project_to_triangle(new_p)
                    if np.allclose(new_p, pts[i]):
                        continue
                    
                    old_p = pts[i]
                    pts[i] = new_p
                    val = -_min_triangle_area(pts)
                    if val < best_local_val:
                        best_local_val = val
                        best_p = new_p.copy()
                        improved = True
                    pts[i] = old_p
            
            if not np.allclose(pts[i], best_p):
                pts[i] = best_p
                improved = True
        
        if not improved:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Initial setup: use a structured initial configuration
    # Start with points roughly evenly distributed in the triangle
    h = np.sqrt(3) / 2
    
    # Create a grid of candidate positions and pick a good initial set
    # Use barycentric coordinates for even distribution
    def _initial_points():
        """Generate initial points using a structured pattern."""
        pts = []
        # Add vertices
        pts.append([0.0, 0.0])
        pts.append([1.0, 0.0])
        pts.append([0.5, h])
        # Add midpoints of edges
        pts.append([0.5, 0.0])
        pts.append([0.25, h/2])
        pts.append([0.75, h/2])
        # Add center
        pts.append([0.5, h/3])
        # Add a few more interior points
        pts.append([0.3, h/6])
        pts.append([0.7, h/6])
        pts.append([0.5, h/2])
        return np.array(pts[:N])
    
    best_pts = _initial_points()
    best_val = _min_triangle_area(best_pts)
    
    # Try multiple random restarts with differential evolution
    rng = np.random.RandomState(42)
    
    n_restarts = 0
    while time.time() < deadline - 30:  # Leave 30s for final polish
        n_restarts += 1
        
        # Generate random initial points inside the triangle
        def _random_point_in_triangle():
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = h * v
            return np.array([x, y])
        
        # Random initialization
        init_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        
        # Use differential evolution with bounds
        # Bounds: x in [0, 1], y in [0, h]
        bounds = [(0, 1), (0, h)] * N
        
        # Run differential evolution with time-limited approach
        def _de_objective(params):
            if time.time() > deadline - 35:
                return 1e10
            pts = params.reshape(N, 2)
            # Project to triangle
            for i in range(N):
                pts[i] = _project_to_triangle(pts[i])
            return _objective_with_penalty(params)
        
        try:
            result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                seed=n_restarts,
                update_interval=1,
                polish=False,
                workers=1
            )
        except Exception:
            continue
        
        if result.success or result.fun < 1e9:
            de_pts = result.x.reshape(N, 2)
            # Project to triangle
            for i in range(N):
                de_pts[i] = _project_to_triangle(de_pts[i])
            
            # Coordinate descent polish
            de_pts = _coordinate_descent(de_pts, max_iter=200, time_limit=10, start_time=time.time())
            
            de_val = _min_triangle_area(de_pts)
            if de_val > best_val:
                best_val = de_val
                best_pts = de_pts.copy()
        
        # Also try coordinate descent on the random init
        if time.time() < deadline - 30:
            cd_pts = _coordinate_descent(init_pts, max_iter=100, time_limit=5, start_time=time.time())
            cd_val = _min_triangle_area(cd_pts)
            if cd_val > best_val:
                best_val = cd_val
                best_pts = cd_pts.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        best_pts = _coordinate_descent(best_pts, max_iter=500, tol=1e-10, time_limit=20, start_time=time.time())
    
    # Ensure all points are strictly inside or on the triangle
    for i in range(N):
        best_pts[i] = _project_to_triangle(best_pts[i])
    
    return best_pts
# EVOLVE-BLOCK-END
