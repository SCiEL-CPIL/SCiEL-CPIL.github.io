"""Heilbronn triangle problem: maximize minimum area among 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # area of equilateral triangle with side 1

def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _inside_triangle(p, tri_verts):
    """Check if point p is inside or on the boundary of triangle defined by tri_verts."""
    # Use barycentric coordinates
    (x0, y0), (x1, y1), (x2, y2) = tri_verts
    px, py = p
    # Vectors
    v0 = (x1 - x0, y1 - y0)
    v1 = (x2 - x0, y2 - y0)
    v2 = (px - x0, py - y0)
    # Dot products
    d00 = v0[0]*v0[0] + v0[1]*v0[1]
    d01 = v0[0]*v1[0] + v0[1]*v1[1]
    d02 = v0[0]*v2[0] + v0[1]*v2[1]
    d11 = v1[0]*v1[0] + v1[1]*v1[1]
    d12 = v1[0]*v2[0] + v1[1]*v2[1]
    # Barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    tri_verts = [(0, 0), (1, 0), (0.5, SQRT3 / 2)]
    clipped = []
    for p in points:
        if _inside_triangle(p, tri_verts):
            clipped.append(p)
        else:
            # Project to nearest point on triangle boundary
            # Simple approach: scale towards centroid
            cx, cy = 0.5, SQRT3 / 6
            t = 1.0
            # Binary search for largest t such that p + t*(c - p) is inside
            for _ in range(50):
                mid = (t + 0) / 2
                test_p = (p[0] + mid * (cx - p[0]), p[1] + mid * (cy - p[1]))
                if _inside_triangle(test_p, tri_verts):
                    t = mid
                else:
                    t = (t + 0) / 2  # This is wrong, let me fix
            # Actually let's do it properly
            t = 1.0
            for _ in range(60):
                test_p = (p[0] * (1 - t) + cx * t, p[1] * (1 - t) + cy * t)
                if _inside_triangle(test_p, tri_verts):
                    t = t / 2
                else:
                    t = t * 1.5
            clipped.append((p[0] * (1 - t) + cx * t, p[1] * (1 - t) + cy * t))
    return np.array(clipped)

def _objective(x, target_t=None):
    """Objective: negative of minimum triangle area (to minimize)."""
    points = x.reshape(N, 2)
    # Penalty for points outside triangle
    tri_verts = [(0, 0), (1, 0), (0.5, SQRT3 / 2)]
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p, tri_verts):
            # Distance outside
            cx, cy = 0.5, SQRT3 / 6
            dx, dy = p[0] - cx, p[1] - cy
            penalty += (dx*dx + dy*dy) * 100
    if penalty > 0:
        return penalty
    min_a = _min_triangle_area(points)
    return -min_a

def _construct_initial():
    """Create initial configuration using a structured approach."""
    # Start with a grid-like arrangement inside the triangle
    points = []
    # Use a few strategic points
    # Corners and center
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, SQRT3 / 2])
    points.append([0.5, SQRT3 / 6])  # centroid
    # Add more points in a pattern
    # Let's use a more systematic approach: place points at vertices of a smaller equilateral triangle pattern
    # For 11 points, use a hexagonal-like arrangement
    # Actually, let's just use random initialization with multiple restarts
    
    rng = np.random.RandomState(42)
    best_points = None
    best_score = -1
    
    for attempt in range(20):
        # Generate random points inside triangle
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                pts.append([u + 0.5 * v, SQRT3 / 2 * v])
        pts = np.array(pts)
        score = _min_triangle_area(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    return best_points

def construct():
    """Main entry point: find 11 points maximizing minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initial configuration
    points = _construct_initial()
    best_points = points.copy()
    best_score = _min_triangle_area(best_points)
    
    # Optimize using Nelder-Mead
    x0 = best_points.flatten()
    
    # Try multiple random restarts
    rng = np.random.RandomState(123)
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Perturb best solution
            x0 = best_points.flatten() + rng.normal(0, 0.05, size=N * 2)
            # Clip to triangle
            pts = x0.reshape(N, 2)
            pts = _clip_to_triangle(pts)
            x0 = pts.flatten()
        
        # Run Nelder-Mead optimization
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if time.time() > deadline:
            break
        
        opt_points = result.x.reshape(N, 2)
        # Clip to triangle
        opt_points = _clip_to_triangle(opt_points)
        
        score = _min_triangle_area(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
    
    # Final refinement: try to improve by local search on the best solution
    if time.time() < deadline - 5:
        # Do a fine-grained local search
        x0 = best_points.flatten()
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        opt_points = result.x.reshape(N, 2)
        opt_points = _clip_to_triangle(opt_points)
        score = _min_triangle_area(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
    
    return best_points
