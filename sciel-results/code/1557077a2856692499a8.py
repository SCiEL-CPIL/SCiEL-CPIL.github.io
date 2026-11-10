# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    normals = []
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        normals.append(normal)
    return np.array(normals)

def _check_separation(verts1, verts2):
    n1 = _edge_normals(verts1)
    n2 = _edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _check_inside(hex_verts, outer_verts):
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        for p in hex_verts:
            cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy = outer_data[0], outer_data[1]
    for i in range(11):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        verts_shifted = verts - np.array([ox, oy])
        for j in range(6):
            p1 = verts_shifted[j]
            p2 = verts_shifted[(j + 1) % 6]
            edge_len = np.linalg.norm(p2 - p1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        inner_verts.append(verts)
    
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], side, outer_data[2])
    for i in range(11):
        if not _check_inside(inner_verts[i], outer_verts):
            return False
    
    return True

def _objective(params):
    side = params[0]
    if side < np.sqrt(11) - 0.5:
        return 1e10
    
    inner_data = params[4:].reshape(11, 3)
    outer_verts = _hex_vertices(params[1], params[2], side, params[3])
    
    penalty = 0.0
    inner_verts = []
    for i in range(11):
        verts = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts.append(verts)
    
    for i in range(11):
        for j in range(i + 1, 11):
            v1, v2 = inner_verts[i], inner_verts[j]
            n1 = _edge_normals(v1)
            n2 = _edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            max_gap = -np.inf
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                max_gap = max(max_gap, gap)
            if max_gap < 1e-6:
                penalty += max(0, (1e-6 - max_gap)) ** 2
    
    for i in range(11):
        hv = inner_verts[i]
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k + 1) % 6]
            edge = p2 - p1
            for p in hv:
                cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                if cross < 0:
                    penalty += max(0, -cross) ** 2
    
    return side + 50 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620
    
    s3 = np.sqrt(3)
    
    # Flower pattern initial guess
    inner_init = np.array([
        [0, 0, 0],
        [s3, 0, 0],
        [s3/2, s3*np.sqrt(3)/2, 0],
        [-s3/2, s3*np.sqrt(3)/2, 0],
        [-s3, 0, 0],
        [-s3/2, -s3*np.sqrt(3)/2, 0],
        [s3/2, -s3*np.sqrt(3)/2, 0],
        [s3*1.5, s3*np.sqrt(3)/2, 0],
        [0, s3*np.sqrt(3), 0],
        [-s3*1.5, s3*np.sqrt(3)/2, 0],
        [s3*1.5, -s3*np.sqrt(3)/2, 0],
    ])
    
    outer_init = np.array([0, 0, 0])
    side_init = 4.0
    
    best_inner = inner_init.copy()
    best_outer = outer_init.copy()
    best_side = side_init
    
    if _is_valid(best_inner, best_outer, best_side):
        pass
    else:
        # Try to find a valid starting point
        for s in np.linspace(4.0, 6.0, 5):
            if _is_valid(best_inner, best_outer, s):
                best_side = s
                break
    
    best_score = 1.0 / best_side
    
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            inner_try = inner_init.copy()
            side_try = best_side
        else:
            inner_try = inner_init + np.random.randn(11, 3) * 0.3
            inner_try[:, 2] = np.random.uniform(-30, 30, 11)
            side_try = best_side + np.random.uniform(-0.3, 0.3)
        
        params = np.concatenate([[side_try], outer_init, inner_try.flatten()])
        
        try:
            result = minimize(_objective, params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
        except:
            continue
        
        if time.time() > deadline - 60:
            break
        
        side_opt = result.x[0]
        inner_opt = result.x[4:].reshape(11, 3)
        outer_opt = result.x[1:4]
        
        if side_opt >= np.sqrt(11) and _is_valid(inner_opt, outer_opt, side_opt):
            if side_opt < best_side:
                best_side = side_opt
                best_inner = inner_opt
                best_outer = outer_opt
                best_score = 1.0 / best_side
    
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
