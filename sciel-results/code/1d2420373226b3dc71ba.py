import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.arange(6) * (np.pi / 3)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _hex_overlap_penalty(verts1, verts2):
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            normals.append(normal)
        return normals

    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_overlap = np.inf
    overlap = False
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        proj2 = np.dot(verts2, normal)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        else:
            overlap_depth = min(max1, max2) - max(min1, min2)
            if overlap_depth < min_overlap:
                min_overlap = overlap_depth
            overlap = True
            
    if overlap:
        return min_overlap ** 2
    else:
        return 0.0

def _min_side_length(inner_hexes):
    max_required_S = 0.0
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    for cx, cy, angle in inner_hexes:
        verts = _hex_vertices(cx, cy, 1.0, angle)
        proj = np.dot(verts, normals.T)
        for k in range(6):
            req_S = (2.0 / np.sqrt(3.0)) * proj[:, k]
            if req_S > max_required_S:
                max_required_S = req_S
    return max_required_S

def _objective(params):
    if time.time() > DEADLINE:
        return 1e10
        
    inner_hexes = []
    for i in range(N):
        cx = params[i * 3]
        cy = params[i * 3 + 1]
        angle = params[i * 3 + 2]
        inner_hexes.append((cx, cy, angle))
        
    S_min = _min_side_length(inner_hexes)
    
    penalty = 0.0
    verts_list = [_hex_vertices(cx, cy, 1.0, angle) for cx, cy, angle in inner_hexes]
    
    for i in range(N):
        for j in range(i + 1, N):
            p = _hex_overlap_penalty(verts_list[i], verts_list[j])
            penalty += p
            if penalty > 1000:
                return S_min + penalty * 1000
                
    return S_min + penalty * 1000

def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * N
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        
        best_params = result.x
        inner = []
        for i in range(N):
            cx = best_params[i * 3]
            cy = best_params[i * 3 + 1]
            angle = best_params[i * 3 + 2]
            inner.append([cx, cy, angle])
            
        S = _min_side_length([(p[0], p[1], p[2]) for p in inner])
        
        return inner, (0.0, 0.0), S, 0.0

    except Exception as e:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
