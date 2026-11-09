import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def _hex_verts(cx, cy, theta_rad):
    """
    Calculate vertices of a unit regular hexagon centered at (cx, cy) 
    with rotation theta_rad.
    """
    angles = np.radians(60 * np.arange(6))
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    
    rx = base_x * c - base_y * s
    ry = base_x * s + base_y * c
    
    return np.column_stack((rx + cx, ry + cy))

def _sat_overlap_depth(h1, h2):
    """
    Calculate minimum penetration depth between two convex polygons using SAT.
    Returns 0 if disjoint.
    """
    min_depth = float('inf')
    
    axes = []
    for h in (h1, h2):
        for i in range(6):
            p1 = h[i]
            p2 = h[(i + 1) % 6]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
            
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
            
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def _compute_side_length_and_penalty(params):
    """
    Compute the required outer side length and the total overlap penalty.
    params: array of shape (36,) -> [x0, y0, t0, x1, y1, t1, ...]
    Returns: (side_length, overlap_penalty)
    """
    outer_normals_angles = np.radians(30 + 60 * np.arange(6))
    outer_normals = np.column_stack((np.cos(outer_normals_angles), np.sin(outer_normals_angles)))
    
    max_proj = 0.0
    vertices_list = []
    
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        theta = params[3*i+2]
        
        verts = _hex_verts(cx, cy, theta)
        vertices_list.append(verts)
        
        projs = verts @ outer_normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    side_length = 2.0 * max_proj / math.sqrt(3.0)
    
    total_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(vertices_list[i], vertices_list[j])
            if depth > 0:
                total_overlap += depth
                
    return side_length, total_overlap

def _objective(params):
    """
    Objective function for optimization.
    """
    side_length, overlap = _compute_side_length_and_penalty(params)
    
    if overlap > 1e-6:
        return side_length + 100.0 * overlap
    else:
        return side_length

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    init_positions = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            init_positions.append((x, y))
            
    xs = [p[0] for p in init_positions]
    ys = [p[1] for p in init_positions]
    cx_avg = sum(xs) / len(xs)
    cy_avg = sum(ys) / len(ys)
    
    init_params = []
    for x, y in init_positions:
        init_params.extend([x - cx_avg, y - cy_avg, 0.0])
    init_params = np.array(init_params)
    
    best_result = None
    best_score = float('inf')
    
    bounds = [(-4.0, 4.0), (-4.0, 4.0), (0.0, 60.0)] * N
    
    try:
        if time.time() < deadline - 20.0:
            de_result = differential_evolution(
                _objective,
                bounds,
                init='latinhypercube',
                maxiter=100,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,
                update_interval=10
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x
                
    except Exception:
        pass
        
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            polish_result = minimize(
                _objective,
                best_result,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass
            
    if best_result is None or best_score > 10.0:
        try:
            if time.time() < deadline - 5.0:
                local_result = minimize(
                    _objective,
                    init_params,
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                if local_result.fun < best_score:
                    best_score = local_result.fun
                    best_result = local_result.x
        except Exception:
            pass
            
    if best_result is None:
        best_result = init_params
        
    side_length, overlap = _compute_side_length_and_penalty(best_result)
    
    inner = []
    for i in range(N):
        x = float(best_result[3*i])
        y = float(best_result[3*i+1])
        angle_deg = float(best_result[3*i+2])
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), float(side_length), 0.0
