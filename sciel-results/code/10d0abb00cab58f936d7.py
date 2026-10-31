"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay
import sys

N = 11

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, vertices):
    """Check if point p is inside or on the boundary of triangle with vertices."""
    # Use barycentric coordinates
    x, y = p
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    
    # Vectors
    v0 = np.array([x2 - x0, y2 - y0])
    v1 = np.array([x1 - x0, y1 - y0])
    v2 = np.array([x - x0, y - y0])
    
    # Dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    # Barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    a = (dot11 * dot02 - dot01 * dot12) / denom
    b = (dot00 * dot12 - dot01 * dot02) / denom
    c = 1.0 - a - b
    
    # Point is inside if all barycentric coordinates are non-negative
    return a >= -1e-9 and b >= -1e-9 and c >= -1e-9

def _clamp_to_triangle(p, vertices):
    """Clamp point p to be inside the triangle with given vertices."""
    # Simple approach: if outside, project to nearest edge
    # For robustness, we'll use a simple rejection or projection
    # Here we use a simple method: if outside, move toward centroid
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Check if inside
    if _is_inside_triangle(p, vertices):
        return np.array([x, y])
    
    # Project to nearest edge
    # Edge 1: from (0,0) to (1,0) - y >= 0
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2)
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0)
    
    best_point = np.array([0.5, 0.5 * np.sqrt(3) / 2])  # centroid as fallback
    best_dist = float('inf')
    
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        # Project p onto line segment p1-p2
        v = p2 - p1
        w = p - p1
        c1 = np.dot(w, v)
        if c1 < 0:
            proj = p1
        else:
            c2 = np.dot(v, v)
            if c2 < 1e-12:
                proj = p1
            else:
                t = c1 / c2
                if t > 1:
                    proj = p2
                else:
                    proj = p1 + t * v
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best_point = proj
    
    return best_point

def _objective(x, vertices):
    """Objective function to minimize (negative of min triangle area)."""
    points = np.array(x).reshape(-1, 2)
    # Clamp points to triangle
    points = np.array([_clamp_to_triangle(p, vertices) for p in points])
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Triangle vertices
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    centroid = np.mean(vertices, axis=0)
    
    # Initial guess: grid-like distribution
    def _initial_points():
        # Use a structured initial guess
        pts = []
        # Corners and midpoints plus center
        pts.append(vertices[0] + 0.05 * (centroid - vertices[0]))
        pts.append(vertices[1] + 0.05 * (centroid - vertices[1]))
        pts.append(vertices[2] + 0.05 * (centroid - vertices[2]))
        # Edge midpoints (slightly inward)
        for i in range(3):
            mid = 0.5 * (vertices[i] + vertices[(i + 1) % 3])
            mid = 0.9 * mid + 0.1 * centroid
            pts.append(mid)
        # Center and some interior points
        pts.append(centroid)
        # Add 4 more interior points
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        for a in angles:
            r = 0.2
            p = centroid + r * np.array([np.cos(a), np.sin(a)])
            pts.append(_clamp_to_triangle(p, vertices))
        return np.array(pts)
    
    best_points = _initial_points()
    best_score = _min_triangle_area(best_points)
    
    # Try differential evolution with multiple restarts
    bounds = [(0.0, 1.0), (0.0, 0.8660254037844386)] * N
    
    # Use a better initial population
    def _random_population():
        pop = []
        for _ in range(50):
            pts = []
            for _ in range(N):
                u = random.random()
                v = random.random()
                while u + v > 1.0:
                    u = random.random()
                    v = random.random()
                p = u * vertices[1] + v * vertices[2]
                pts.append(p)
            pts = np.array(pts)
            score = _min_triangle_area(pts)
            pop.append((score, pts))
        return pop
    
    # Run differential evolution with time check
    def _run_de(bounds, init, maxiter, tol, seed):
        result = differential_evolution(
            _objective,
            bounds,
            args=(vertices,),
            init=init,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            popsize=15,
            mutation=(0.3, 0.8),
            recombination=0.8,
            polish=False,
            callback=None
        )
        return result
    
    # First, do a quick DE run
    try:
        result = _run_de(bounds, 'latinhypercube', maxiter=20, tol=1e-6, seed=42)
        if result.success or result.fun < 0:
            pts = result.x.reshape(-1, 2)
            pts = np.array([_clamp_to_triangle(p, vertices) for p in pts])
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return best_points
    
    # Try multiple local optimizations from different starting points
    starting_points = []
    # Add the current best
    starting_points.append(best_points.copy())
    
    # Add some random starts
    for _ in range(10):
        pts = []
        for _ in range(N):
            u = random.random()
            v = random.random()
            while u + v > 1.0:
                u = random.random()
                v = random.random()
            p = u * vertices[1] + v * vertices[2]
            pts.append(p)
        pts = np.array(pts)
        pts = np.array([_clamp_to_triangle(p, vertices) for p in pts])
        starting_points.append(pts)
    
    # Add structured starts
    for r in [0.1, 0.3, 0.5]:
        pts = []
        # Place points on concentric "rings"
        # Center
        pts.append(centroid)
        # Ring 1
        for a in np.linspace(0, 2 * np.pi, 5, endpoint=False):
            p = centroid + r * 0.5 * np.array([np.cos(a), np.sin(a)])
            pts.append(_clamp_to_triangle(p, vertices))
        # Ring 2
        for a in np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5:
            p = centroid + r * np.array([np.cos(a), np.sin(a)])
            pts.append(_clamp_to_triangle(p, vertices))
        pts = np.array(pts[:N])
        starting_points.append(pts)
    
    # Local optimization using Nelder-Mead or L-BFGS-B
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _objective,
                start.flatten(),
                args=(vertices,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            pts = result.x.reshape(-1, 2)
            pts = np.array([_clamp_to_triangle(p, vertices) for p in pts])
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
