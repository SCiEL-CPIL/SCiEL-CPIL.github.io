# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _check_separation(hex1, hex2):
    def get_normals(vertices):
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            normals[i] = n
        return normals

    n1 = get_normals(hex1)
    n2 = get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms

    for n in all_normals:
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _required_outer_size(all_vertices, outer_center, outer_angle_deg):
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    rel_vertices = all_vertices - outer_center
    proj = rel_vertices @ normals.T
    max_proj = np.max(proj)
    if max_proj < 0:
        return 0.0
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = []
    for v in inner_verts:
        shifted_inner.append(v - np.array([ox, oy]))

    for iv in shifted_inner:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                # 2D cross product: (p2-p1) x (p-p1)
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False

    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590.0

    h = np.sqrt(3)
    
    # Configuration 1: Standard Flower + 4 in gaps
    centers1 = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [2*h*np.cos(np.deg2rad(30)), 2*h*np.sin(np.deg2rad(30))],
        [2*h*np.cos(np.deg2rad(150)), 2*h*np.sin(np.deg2rad(150))],
        [2*h*np.cos(np.deg2rad(210)), 2*h*np.sin(np.deg2rad(210))],
        [2*h*np.cos(np.deg2rad(330)), 2*h*np.sin(np.deg2rad(330))],
    ])
    
    # Configuration 2: Loose grid
    sp = 2.1
    centers2 = np.array([
        [0, 0], [-sp, 0], [sp, 0],
        [-sp/2, h], [sp/2, h], [-3*sp/2, h], [3*sp/2, h],
        [0, 2*h], [-sp, 2*h], [sp, 2*h],
        [-sp/2, 3*h], [sp/2, 3*h]
    ])
    
    configs = [centers1, centers2]
    
    best_S = float('inf')
    best_inner = None
    best_outer = None
    
    for centers in configs:
        if time.time() > deadline: break
        
        inner_verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
        valid_internal = True
        for i in range(11):
            for j in range(i + 1, 11):
                if not _check_separation(inner_verts[i], inner_verts[j]):
                    valid_internal = False
                    break
            if not valid_internal: break
        
        if not valid_internal:
            continue
            
        all_inner_vertices = np.vstack(inner_verts)
        
        def obj_outer(params):
            ox, oy, oang = params
            S = _required_outer_size(all_inner_vertices, np.array([ox, oy]), oang)
            return S
        
        centroid = np.mean(centers, axis=0)
        best_local_S = float('inf')
        best_local_params = None
        
        for ang0 in [0, 30, 60, 90, 120, 150]:
            x0 = [centroid[0], centroid[1], float(ang0)]
            try:
                res = minimize(obj_outer, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x
            except Exception:
                pass
            
        if best_local_params is not None:
            ox, oy, oang = best_local_params
            S = best_local_S
            if S < best_S:
                best_S = S
                best_inner = centers.copy()
                best_outer = np.array([ox, oy, oang])
                
    if best_inner is None:
        best_inner = centers1
        centroid = np.mean(best_inner, axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        inner_verts = [_hex_vertices(best_inner[i, 0], best_inner[i, 1], 1.0, 0.0) for i in range(11)]
        all_inner_vertices = np.vstack(inner_verts)
        best_S = _required_outer_size(all_inner_vertices, np.array([centroid[0], centroid[1]]), 0.0)
        
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
        
    outer_hex_data = best_outer
    
    if not _validate(inner_hex_data, outer_hex_data, best_S):
        best_S += 1e-4
        if not _validate(inner_hex_data, outer_hex_data, best_S):
            best_S = 10.0
            inner_hex_data = np.array([
                [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0]
            ])
            outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
