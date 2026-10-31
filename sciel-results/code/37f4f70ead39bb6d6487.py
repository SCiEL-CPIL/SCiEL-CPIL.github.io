import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

def point_in_triangle(p, v0, v1, v2, eps=1e-9):
    # Check if point p is inside triangle v0, v1, v2
    # Using barycentric coordinates
    # p = u*v0 + v*v1 + w*v2, u+v+w=1
    # Solve for u, v, w
    # v1 - v0, v2 - v0
    p1 = v1 - v0
    p2 = v2 - v0
    p3 = p - v0
    
    # Cross products
    d1 = p1[0] * p2[1] - p1[1] * p2[0]
    if np.abs(d1) < 1e-12:
        return False # Degenerate triangle (should not happen)
        
    u = (p3[0] * p2[1] - p3[1] * p2[0]) / d1
    v = (p1[0] * p3[1] - p1[1] * p3[0]) / d1
    w = 1 - u - v
    
    return u >= -eps and v >= -eps and w >= -eps

def min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate areas for all triples
    # Vectorized approach
    # i, j, k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # We can compute all pairwise differences first? No, triples are needed.
    # For N=11, 165 triangles. Simple loop is fast enough in Python if not too many iterations?
    # 165 * (few ops) is very fast.
    
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p_i = points[i]
                p_j = points[j]
                p_k = points[k]
                # Cross product of (p_j - p_i) and (p_k - p_i)
                dx1 = p_j[0] - p_i[0]
                dy1 = p_j[1] - p_i[1]
                dx2 = p_k[0] - p_i[0]
                dy2 = p_k[1] - p_i[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    # x is a flattened array of shape (2*N,)
    points = x.reshape(N, 2)
    
    # Check if all points are inside the triangle
    # Penalty if outside
    for p in points:
        if not point_in_triangle(p, V0, V1, V2):
            # Return a large negative value for maximization (or large positive for minimization)
            # We are minimizing -area + penalty.
            # So if outside, we want to minimize a large positive number?
            # Let's return a large positive penalty so the minimizer moves away.
            return 1e6
    
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    best_points = None
    best_area = -1.0
    
    # Initial configurations
    # 1. Random uniform
    # 2. Grid-like
    # 3. Vertices + Midpoints + Center + Others
    
    configs = []
    
    # Config 1: Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Sample in bounding box and filter? Or rejection sampling.
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                pts.append(p)
        configs.append(np.array(pts).flatten())
        
    # Config 2: Structured
    # Vertices
    p0 = V0
    p1 = V1
    p2 = V2
    # Midpoints
    m01 = (V0 + V1) / 2
    m12 = (V1 + V2) / 2
    m20 = (V2 + V0) / 2
    # Center
    c = (V0 + V1 + V2) / 3
    
    # We have 7 points. Need 4 more.
    # Place them along the medians?
    # Or in a smaller grid inside?
    # Let's just add some random points near the center for the remaining 4
    remaining = []
    for _ in range(4):
        u, v = rng.random(2)
        if u + v < 1:
            p = c + 0.2 * (V0 + u * (V1 - V0) + v * (V2 - V0) - c) # Scale down around center?
            # Simpler: sample in triangle, shrink towards center
            p_sample = V0 + u * (V1 - V0) + v * (V2 - V0)
            p_shrunk = c + 0.5 * (p_sample - c)
            remaining.append(p_shrunk)
            
    structured = np.array([p0, p1, p2, m01, m12, m20, c] + remaining)
    configs.append(structured.flatten())
    
    # Optimization
    for i, x0 in enumerate(configs):
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead for robustness
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < objective(x0):
                x_opt = result.x
                area_opt = -result.fun
                
                # Verify validity
                pts_opt = x_opt.reshape(N, 2)
                valid = all(point_in_triangle(p, V0, V1, V2) for p in pts_opt)
                
                if valid:
                    # Re-calculate area to be sure
                    area_check = min_triangle_area(pts_opt)
                    if area_check > best_area:
                        best_area = area_check
                        best_points = pts_opt
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to the first config
        best_points = configs[0].reshape(N, 2)
        
    return best_points
