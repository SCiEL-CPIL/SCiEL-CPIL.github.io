"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, t1, t2, t3, tol=1e-9):
    """Check if point p is inside or on the boundary of triangle t1,t2,t3."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, t1, t2)
    d2 = sign(p, t2, t3)
    d3 = sign(p, t3, t1)
    
    has_neg = (d1 < -tol) or (d2 < -tol) or (d3 < -tol)
    has_pos = (d1 > tol) or (d2 > tol) or (d3 > tol)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p, t1, t2, t3):
    """Project point p onto the triangle t1,t2,t3 if it's outside."""
    if _is_inside_triangle(p, t1, t2, t3):
        return p
    
    # Find the closest point on the triangle boundary
    # Check edges
    best = None
    best_dist = float('inf')
    
    edges = [(t1, t2), (t2, t3), (t3, t1)]
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Define the equilateral triangle vertices
    h = np.sqrt(3) / 2
    T1 = np.array([0.0, 0.0])
    T2 = np.array([1.0, 0.0])
    T3 = np.array([0.5, h])
    
    def objective(x):
        """Negative of min triangle area (we minimize this)."""
        points = x.reshape(-1, 2)
        # Project points back into triangle
        for i in range(len(points)):
            points[i] = _project_to_triangle(points[i], T1, T2, T3)
        
        min_area = _min_triangle_area(points)
        return -min_area
    
    def gradient(x):
        """Approximate gradient using finite differences."""
        points = x.reshape(-1, 2)
        min_area = _min_triangle_area(points)
        grad = np.zeros_like(x)
        eps = 1e-5
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            points_plus = x_plus.reshape(-1, 2)
            for j in range(len(points_plus)):
                points_plus[j] = _project_to_triangle(points_plus[j], T1, T2, T3)
            min_area_plus = _min_triangle_area(points_plus)
            grad[i] = (min_area_plus - min_area) / eps
        return -grad  # We're minimizing -min_area
    
    # Initial configuration: spread points evenly
    # Use a grid-like initial placement
    def initial_guess():
        points = []
        # Place points in a rough grid
        rows = 3
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= N:
                    break
                # Parametric coordinates within triangle
                u = (j + 0.5) / cols
                v = (i + 0.5) / rows
                if u + v < 1.0:
                    x = u + 0.5 * v
                    y = h * v
                    points.append([x, y])
        # Add remaining points at random positions inside triangle
        while len(points) < N:
            u, v = np.random.random(2)
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                points.append([x, y])
        return np.array(points[:N]).flatten()
    
    best_points = None
    best_score = -float('inf')
    
    # Multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        x0 = initial_guess()
        
        # Add small random perturbation
        x0 = x0 + 0.05 * np.random.randn(len(x0))
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = result.x
            points_opt = x_opt.reshape(-1, 2)
            for i in range(len(points_opt)):
                points_opt[i] = _project_to_triangle(points_opt[i], T1, T2, T3)
            
            score = _min_triangle_area(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            continue
    
    # If no valid points found, return a default configuration
    if best_points is None:
        # Fallback: simple grid
        points = []
        for i in range(3):
            for j in range(4):
                if len(points) >= N:
                    break
                u = (j + 0.5) / 4
                v = (i + 0.5) / 3
                if u + v < 0.95:
                    x = u + 0.5 * v
                    y = h * v
                    points.append([x, y])
        while len(points) < N:
            u, v = np.random.random(2)
            if u + v < 0.95:
                x = u + 0.5 * v
                y = h * v
                points.append([x, y])
        best_points = np.array(points[:N])
    
    return best_points

# EVOLVE-BLOCK-END
