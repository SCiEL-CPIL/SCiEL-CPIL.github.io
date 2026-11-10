import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(ex, ey)
        if length > 0:
            normals[i] = np.array([ey / length, -ex / length])
    return normals

def _separation_violation(verts1, verts2, eps=1e-6):
    """Return 0 if separated, else positive penalty."""
    normals1 = _edge_normals(verts1)
    normals2 = _edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - eps or max2 < min1 - eps:
            return 0.0
    # Not separated: return a penalty based on minimum gap (negative)
    # Compute the minimum over all axes of the gap (which is negative if overlapping)
    min_gap = np.inf
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = min(min2 - max1, max1 - max2)  # negative if overlapping
        if gap < min_gap:
            min_gap = gap
    return max(0, -min_gap) * 100 + 10  # penalty

def _containment_violation(inner_verts, outer_verts, eps=1e-6):
    """Return penalty for vertices outside outer hexagon."""
    penalty = 0.0
    for p in inner_verts:
        for i in range(len(outer_verts)):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % len(outer_verts)]
            edge = p2 - p1
            cross = np.cross(edge, p - p1)
            if cross < -eps:
                penalty += abs(cross) * 10
    return penalty

def _objective(params, n_inner=11):
    s = params[0]
    if s < 1e-6:
        return 1e10
    centers = params[1:].reshape(n_inner, 2)
    
    inner_verts_list = [_hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    
    penalty = 0.0
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            penalty += _separation_violation(inner_verts_list[i], inner_verts_list[j])
        penalty += _containment_violation(inner_verts_list[i], outer_verts)
    
    return s + penalty

def hexagon_packing_11():
    n = 11
    sqrt3 = np.sqrt(3)
    
    centers = np.zeros((n, 2))
    centers[0] = [0, 0]
    for i in range(1, 7):
        angle = np.radians(60 * (i - 1))
        centers[i] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    for i in range(7, 11):
        angle = np.radians(30 + 60 * (i - 7))
        centers[i] = [2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)]
    
    max_r = np.max(np.linalg.norm(centers, axis=1)) + 1.0
    s_init = max_r / (np.sqrt(3) / 2)
    
    params0 = np.concatenate([[s_init], centers.flatten()])
    
    result = minimize(_objective, params0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    params_opt = result.x
    s_opt = max(params_opt[0], np.sqrt(11))
    centers_opt = params_opt[1:].reshape(n, 2)
    
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = [centers_opt[i, 0], centers_opt[i, 1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(s_opt)
