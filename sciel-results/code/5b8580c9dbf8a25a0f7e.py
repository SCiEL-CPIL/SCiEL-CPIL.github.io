import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, eps=1e-9):
    denom = np.sqrt(3) / 2
    v = (p[0] * V2[1] - p[1] * V2[0]) / denom
    w = (p[1] * V1[0] - p[0] * V1[1]) / denom
    u = 1 - v - w
    return (u >= -eps) and (v >= -eps) and (w >= -eps)

def calculate_min_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def barycentric_coords(p):
    """Return (u, v, w) barycentric coordinates for point p."""
    denom = np.sqrt(3) / 2
    v = (p[0] * V2[1] - p[1] * V2[0]) / denom
    w = (p[1] * V1[0] - p[0] * V1[1]) / denom
    u = 1 - v - w
    return u, v, w

def clamp_to_triangle(p):
    """Project point p to the nearest point inside the equilateral triangle."""
    u, v, w = barycentric_coords(p)
    if u >= 0 and v >= 0 and w >= 0:
        return p
    # If outside, find the closest boundary point
    # Simple approach: scale toward centroid
    centroid = (V0 + V1 + V2) / 3
    # Binary search for the largest t in [0,1] such that centroid + t*(p - centroid) is inside
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        test_p = centroid + mid * (p - centroid)
        tu, tv, tw = barycentric_coords(test_p)
        if tu >= -1e-10 and tv >= -1e-10 and tw >= -1e-10:
            lo = mid
        else:
            hi = mid
    return centroid + lo * (p - centroid)

def objective_and_penalty(x, points_shape):
    """
    x is a flattened array of 2*N coordinates.
    Returns negative min area + penalty for being outside triangle.
    """
    points = x.reshape(points_shape)
    min_area = calculate_min_area(points)
    
    penalty = 0.0
    for p in points:
        u, v, w = barycentric_coords(p)
        if u < 0: penalty += u * u
        if v < 0: penalty += v * v
        if w < 0: penalty += w * w
    
    return -min_area + 10.0 * penalty

def objective_de(x):
    """
    Objective for differential evolution.
    x is a flattened array of 2*N coordinates.
    Points are interpreted in barycentric-like parameterization to ensure feasibility:
    We use the mapping: for each point, generate (u, v) in [0,1]^2 with u+v<=1,
    then p = u*V1 + v*V2.
    So x has 2N elements, pairs (u_i, v_i) with u_i >= 0, v_i >= 0, u_i + v_i <= 1.
    """
    n = len(x) // 2
    points = np.zeros((n, 2))
    for i in range(n):
        u = x[2*i]
        v = x[2*i+1]
        # Clamp to ensure u+v <= 1
        s = u + v
        if s > 1.0:
            u = u / s
            v = v / s
        points[i] = u * V1 + v * V2
    
    min_area = calculate_min_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin within 150s
    
    best_points = None
    best_score = -float('inf')
    
    points_shape = (N, 2)
    
    # Phase 1: Differential Evolution (global search)
    # Parameterize points via (u, v) in [0,1]^2 with u+v <= 1
    # This ensures all points are inside the triangle by construction
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        de_result = differential_evolution(
            objective_de,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,  # We'll do our own polish
            update_interval=1,
            callback=None
        )
        
        if time.time() < deadline - 30:
            # Convert DE result to Cartesian points
            x_de = de_result.x
            de_points = np.zeros((N, 2))
            for i in range(N):
                u = x_de[2*i]
                v = x_de[2*i+1]
                s = u + v
                if s > 1.0:
                    u = u / s
                    v = v / s
                de_points[i] = u * V1 + v * V2
            
            de_score = calculate_min_area(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
            
    except Exception as e:
        pass
    
    # Phase 2: Nelder-Mead refinement from multiple starting points
    # Start from: DE result, structured points, and random points
    
    refinement_starts = []
    
    # Add DE result if available
    if best_points is not None:
        refinement_starts.append(best_points.copy())
    
    # Structured start: vertices, midpoints, center, and symmetric additions
    pts_struct = [V0, V1, V2]
    M01 = (V0 + V1) / 2
    M12 = (V1 + V2) / 2
    M20 = (V2 + V0) / 2
    pts_struct.extend([M01, M12, M20])
    C = (V0 + V1 + V2) / 3
    pts_struct.append(C)
    
    # Add 4 more points in symmetric positions
    # Quarter points along edges
    Q01a = V0 + 0.25 * (V1 - V0)
    Q01b = V0 + 0.75 * (V1 - V0)
    pts_struct.extend([Q01a, Q01b])
    refinement_starts.append(np.array(pts_struct))
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * V1 + v * V2
                pts.append(p)
        refinement_starts.append(np.array(pts))
    
    # Run Nelder-Mead from each start
    for i, initial_pts in enumerate(refinement_starts):
        if time.time() > deadline - 10:
            break
        
        x0 = initial_pts.flatten()
        
        options = {
            'maxiter': 300,
            'xatol': 1e-5,
            'fatol': 1e-5,
            'adaptive': True
        }
        
        try:
            res = minimize(objective_and_penalty, x0, args=(points_shape,), 
                          method='Nelder-Mead', options=options)
            x_opt = res.x
            points_opt = x_opt.reshape(points_shape)
            
            # Clamp any points that might be slightly outside
            for j in range(N):
                points_opt[j] = clamp_to_triangle(points_opt[j])
            
            valid = True
            for p in points_opt:
                if not point_in_triangle(p, eps=1e-4):
                    valid = False
                    break
            
            if valid:
                score = calculate_min_area(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt
                    
        except Exception as e:
            pass
    
    # If no valid point was found, return a structured fallback
    if best_points is None:
        pts_fallback = [V0, V1, V2]
        M01 = (V0 + V1) / 2
        M12 = (V1 + V2) / 2
        M20 = (V2 + V0) / 2
        pts_fallback.extend([M01, M12, M20])
        C = (V0 + V1 + V2) / 3
        pts_fallback.append(C)
        Q01a = V0 + 0.25 * (V1 - V0)
        Q01b = V0 + 0.75 * (V1 - V0)
        pts_fallback.extend([Q01a, Q01b])
        best_points = np.array(pts_fallback)
    
    return best_points
