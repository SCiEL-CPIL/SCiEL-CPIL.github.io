"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11

# Triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_fast(points):
    """Vectorized computation of minimum triangle area."""
    n = len(points)
    # Compute all pairwise differences
    # For each triple (i,j,k), area = 0.5 * |cross(points[j]-points[i], points[k]-points[i])|
    min_area = float('inf')
    for i in range(n):
        # points relative to point i
        rel = points - points[i]  # shape (n, 2)
        # For each j > i, k > j, compute cross product
        for j in range(i + 1, n):
            # cross of rel[j] and rel[k] for k > j
            cross_vals = np.abs(rel[j, 0] * rel[j+1:, 1] - rel[j, 1] * rel[j+1:, 0])
            if len(cross_vals) > 0:
                local_min = np.min(cross_vals)
                if local_min < min_area:
                    min_area = local_min
    return 0.5 * min_area

def _project_to_triangle(points):
    """Project points onto the interior of the equilateral triangle."""
    # The triangle is defined by vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # In barycentric coordinates: point = a*V0 + b*V1 + c*V2 where a+b+c=1, a,b,c >= 0
    # For a point (x,y):
    #   c = y / (sqrt(3)/2)  (since V2 has y = sqrt(3)/2)
    #   b = (2x - 1) * c + ... let me derive properly
    # 
    # Barycentric coordinates:
    # x = b + 0.5*c  => b = x - 0.5*c
    # y = (sqrt(3)/2)*c => c = 2*y/sqrt(3)
    # a = 1 - b - c
    #
    # So: c = 2*y/sqrt(3), b = x - 0.5*c, a = 1 - b - c
    
    result = points.copy()
    sqrt3 = np.sqrt(3)
    
    for idx in range(len(points)):
        x, y = points[idx]
        c = 2 * y / sqrt3
        b = x - 0.5 * c
        a = 1 - b - c
        
        # Check if all barycentric coords are non-negative
        if a < 0 or b < 0 or c < 0:
            # Project: find closest point in the triangle
            # Simple approach: clamp barycentric coordinates and renormalize
            # Better: use projection onto the triangle edges/vertices
            
            # Find the closest point in the triangle using a simple iterative method
            # Or: project onto the edge that is violated
            # For now, use a simple projection: find the closest point on the triangle boundary
            
            # Let's use a more robust approach: minimize distance to triangle
            # The closest point in a triangle to a point can be found by checking:
            # 1. Interior (already checked)
            # 2. Each edge
            # 3. Each vertex
            
            best_dist = float('inf')
            best_pt = points[idx].copy()
            
            # Check vertices
            for v in V:
                d = np.sum((points[idx] - v) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_pt = v.copy()
            
            # Check edges
            edges = [(V[0], V[1]), (V[1], V[2]), (V[2], V[0])]
            for e1, e2 in edges:
                # Project point onto line segment e1-e2
                d = e2 - e1
                t = np.dot(points[idx] - e1, d) / np.dot(d, d)
                t = np.clip(t, 0, 1)
                proj = e1 + t * d
                dist = np.sum((points[idx] - proj) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = proj.copy()
            
            result[idx] = best_pt
        else:
            # Point is inside or on boundary, keep as is but ensure strict interior
            # Add small margin to stay inside
            eps = 1e-6
            if a < eps:
                a = eps
            if b < eps:
                b = eps
            if c < eps:
                c = eps
            # Renormalize
            s = a + b + c
            a, b, c = a/s, b/s, c/s
            result[idx] = a * V[0] + b * V[1] + c * V[2]
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_points = None
    best_area = 0.0
    
    # Generate initial random points
    rng = np.random.RandomState(42)
    
    # Try multiple random restarts with Nelder-Mead optimization
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate random initial points inside the triangle
        points = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            # Convert to barycentric: a = 1-u-v, b = u, c = v
            # Point = a*V0 + b*V1 + c*V2
            a = 1 - u - v
            b = u
            c = v
            # Add small margin
            a = max(a, 0.01)
            b = max(b, 0.01)
            c = max(c, 0.01)
            s = a + b + c
            a, b, c = a/s, b/s, c/s
            pt = a * V[0] + b * V[1] + c * V[2]
            points.append(pt)
        points = np.array(points)
        
        # Objective: maximize min triangle area => minimize negative min area
        def objective(x):
            pts = x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            return -_min_triangle_area_fast(pts)
        
        # Nelder-Mead optimization
        x0 = points.flatten()
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            opt_points = res.x.reshape(N, 2)
            opt_points = _project_to_triangle(opt_points)
            opt_area = _min_triangle_area_fast(opt_points)
            
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    # If we still have time, do a polish pass with tighter tolerances
    if best_points is not None and time.time() < deadline - 5:
        def objective_polish(x):
            pts = x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            return -_min_triangle_area_fast(pts)
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_polish,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            polish_points = res.x.reshape(N, 2)
            polish_points = _project_to_triangle(polish_points)
            polish_area = _min_triangle_area_fast(polish_points)
            
            if polish_area > best_area:
                best_area = polish_area
                best_points = polish_points.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return random points
        points = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            a = 1 - u - v
            b = u
            c = v
            a = max(a, 0.01)
            b = max(b, 0.01)
            c = max(c, 0.01)
            s = a + b + c
            a, b, c = a/s, b/s, c/s
            pt = a * V[0] + b * V[1] + c * V[2]
            points.append(pt)
        best_points = np.array(points)
    
    return best_points
