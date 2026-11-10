# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def polygon_normals(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms

def sat_overlap(proj_a, proj_b):
    min_a, max_a = np.min(proj_a), np.max(proj_a)
    min_b, max_b = np.min(proj_b), np.max(proj_b)
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    return max(0.0, overlap)

def pair_penalty(verts_a, verts_b, normals_a, normals_b):
    penalty = 0.0
    for n in normals_a:
        proj_a = np.dot(verts_a, n)
        proj_b = np.dot(verts_b, n)
        overlap = sat_overlap(proj_a, proj_b)
        if overlap > 0:
            penalty += overlap * overlap
    for n in normals_b:
        proj_a = np.dot(verts_a, n)
        proj_b = np.dot(verts_b, n)
        overlap = sat_overlap(proj_a, proj_b)
        if overlap > 0:
            penalty += overlap * overlap
    return penalty

def outer_penalty(verts_inner, s_outer):
    verts_outer = get_hex_vertices(0, 0, s_outer, 0)
    penalty = 0.0
    for i in range(6):
        p1 = verts_outer[i]
        p2 = verts_outer[(i+1)%6]
        edge_vec = p2 - p1
        n_out = np.array([edge_vec[1], -edge_vec[0]])
        n_out = n_out / np.linalg.norm(n_out)
        n_in = -n_out
        d_edge = np.dot(p1, n_in)
        projs = np.dot(verts_inner, n_in)
        violations = projs - d_edge
        max_violation = np.max(violations)
        if max_violation > 0:
            penalty += max_violation * max_violation
    return penalty

def cost_function(params):
    inner_params = params[:-1]
    s = params[-1]
    inner_hexes = inner_params.reshape(12, 3)
    
    verts_list = []
    normals_list = []
    for i in range(12):
        cx, cy, angle = inner_hexes[i]
        verts = get_hex_vertices(cx, cy, 1.0, angle)
        normals = polygon_normals(verts)
        verts_list.append(verts)
        normals_list.append(normals)
        
    total_penalty = 0.0
    for i in range(12):
        total_penalty += outer_penalty(verts_list[i], s)
    for i in range(12):
        for j in range(i+1, 12):
            total_penalty += pair_penalty(verts_list[i], verts_list[j], normals_list[i], normals_list[j])
            
    return 100.0 * total_penalty + s * s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 500.0
    
    # Initial guess
    centers = []
    centers.append([0.0, 0.0])
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append([x, y])
    for i in range(5):
        angle = np.deg2rad(60 * i)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        centers.append([x, y])
        
    initial_params = []
    for c in centers:
        initial_params.append([c[0], c[1], 0.0])
    max_r = np.max(np.linalg.norm(centers, axis=1))
    s_init = max_r + 1.5
    initial_params.append(s_init)
    initial_params = np.array(initial_params).flatten()
    
    best_params = initial_params
    best_cost = cost_function(best_params)
    
    # Optimization
    options = {
        'maxiter': 1000,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'disp': False
    }
    
    try:
        res = minimize(cost_function, initial_params, method='Powell', options=options)
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
    except Exception:
        pass
        
    if time.time() < deadline - 50:
        try:
            res2 = minimize(cost_function, best_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_params = res2.x
        except Exception:
            pass

    inner_params = best_params[:-1]
    s = best_params[-1]
    inner_hexes = inner_params.reshape(12, 3)
    
    # Validate
    def is_valid(inner, s):
        verts_list = []
        normals_list = []
        for i in range(12):
            cx, cy, angle = inner[i]
            verts = get_hex_vertices(cx, cy, 1.0, angle)
            normals = polygon_normals(verts)
            verts_list.append(verts)
            normals_list.append(normals)
            
        for i in range(12):
            for j in range(i+1, 12):
                p = pair_penalty(verts_list[i], verts_list[j], normals_list[i], normals_list[j])
                if p > 1e-8:
                    return False
        for i in range(12):
            p = outer_penalty(verts_list[i], s)
            if p > 1e-8:
                return False
        return True

    if not is_valid(inner_hexes, s):
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    outer_hex_data = np.array([0, 0, 0])
    return inner_hexes, outer_hex_data, s

# EVOLVE-BLOCK-END
