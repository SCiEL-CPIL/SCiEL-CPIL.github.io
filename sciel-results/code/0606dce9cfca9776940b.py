import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

def hex_vertices(center, angle_deg, side=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + theta
    r = side
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def project_poly(vertices, axis):
    dots = vertices @ axis
    return np.min(dots), np.max(dots)

def min_distance_poly(p1, p2):
    axes = []
    for p in [p1, p2]:
        for i in range(6):
            v1 = p[i]
            v2 = p[(i + 1) % 6]
            edge = v2 - v1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
    
    min_gap = float('inf')
    for axis in axes:
        min1, max1 = project_poly(p1, axis)
        min2, max2 = project_poly(p2, axis)
        
        if max1 <= min2:
            gap = min2 - max1
        elif max2 <= min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

def calculate_required_side(vertices_all):
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    max_proj = 0.0
    for n in normals:
        projs = vertices_all @ n
        max_p = np.max(np.abs(projs))
        if max_p > max_proj:
            max_proj = max_p
            
    return 2.0 * max_proj / np.sqrt(3.0)

def objective(params):
    penalty_weight = 100.0
    positions = []
    angles = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        positions.append((x, y))
        angles.append(a)
        
    all_vertices = []
    hex_verts = []
    for i in range(N):
        v = hex_vertices(positions[i], angles[i])
        hex_verts.append(v)
        all_vertices.append(v)
        
    all_vertices = np.vstack(all_vertices)
    s_req = calculate_required_side(all_vertices)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist_c = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            if dist_c > 4.0:
                continue
            d = min_distance_poly(hex_verts[i], hex_verts[j])
            if d < 0:
                penalty += -d
                
    return s_req + penalty_weight * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    init_params = []
    for u, v in lattice_coords:
        x = np.sqrt(3) * (u + v / 2.0)
        y = 1.5 * v
        angle = 0.0
        init_params.extend([x, y, angle])
        
    init_params = np.array(init_params)
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 360.0))
        
    best_result = init_params
    best_score = objective(init_params)
    
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=30, 
            popsize=10, 
            tol=1e-4
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if time.time() < deadline - 10.0:
        try:
            polish_result = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass

    inner = []
    for i in range(N):
        x, y, a = best_result[3*i:3*i+3]
        inner.append([x, y, a])
        
    positions = []
    angles = []
    for i in range(N):
        x, y, a = best_result[3*i:3*i+3]
        positions.append((x, y))
        angles.append(a)
        
    all_vertices = []
    for i in range(N):
        v = hex_vertices(positions[i], angles[i])
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    s_final = calculate_required_side(all_vertices)
    
    return inner, (0.0, 0.0), s_final, 0.0
