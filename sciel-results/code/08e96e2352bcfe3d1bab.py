# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _sat_overlap_penalty(v1, v2, normals1, normals2, eps=1e-8):
    all_normals = np.vstack([normals1, normals2])
    total_penalty = 0.0
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > 0:
            total_penalty += overlap * overlap
    return total_penalty

def _point_outside_hex_penalty(point, outer_vertices):
    penalty = 0.0
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        to_point = point - p1
        cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross_val < 0:
            penalty += cross_val * cross_val
    return penalty

def _objective(params, s, angle_inner, angle_outer):
    centers = params.reshape(12, 2)
    penalty = 0.0
    
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            penalty += _sat_overlap_penalty(inner_verts[i], inner_verts[j], 
                                            inner_normals[i], inner_normals[j])
    
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    
    for i in range(12):
        for v in inner_verts[i]:
            penalty += _point_outside_hex_penalty(v, outer_verts)
    
    return penalty

def _is_feasible(params, s, angle_inner, angle_outer, tol=1e-6):
    centers = params.reshape(12, 2)
    
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            for n in all_normals:
                p1 = inner_verts[i] @ n
                p2 = inner_verts[j] @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    return False
    
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                to_point = v - p1
                cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
                if cross_val < -tol:
                    return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    best_score = 0.0
    best_result = None
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Configuration 1: 7 center + 5 Sides of Ring 2
    # Sides are at angles 30, 90, 150, 210, 270, 330
    # We pick 5. Let's pick 30, 90, 150, 210, 270.
    # Axial coords for these:
    # 30 deg: (1, 1)
    # 90 deg: (0, 2)
    # 150 deg: (-1, 2)
    # 210 deg: (-1, -1)
    # 270 deg: (0, -2)
    
    centers_r1 = [
        axial_to_cartesian(0, 0),
        axial_to_cartesian(1, 0),
        axial_to_cartesian(1, -1),
        axial_to_cartesian(0, -1),
        axial_to_cartesian(-1, 0),
        axial_to_cartesian(-1, 1),
        axial_to_cartesian(0, 1)
    ]
    
    r2_sides = [
        axial_to_cartesian(1, 1),
        axial_to_cartesian(0, 2),
        axial_to_cartesian(-1, 2),
        axial_to_cartesian(-1, -1),
        axial_to_cartesian(0, -2)
    ]
    
    all_centers = centers_r1 + r2_sides
    x0 = np.array(all_centers).flatten()
    
    angle_inner = 0.0
    angle_outer = 30.0 # Align outer vertices with inner side centers?
    
    # Theoretical min s for this config
    # Max radius is 3 (center) + 1 (vertex) = 4.
    # If aligned, s >= 4.
    # But side constraint might require more.
    s_min_theoretical = 4.0
    
    s_start = s_min_theoretical + 0.5
    
    if _is_feasible(x0, s_start, angle_inner, angle_outer):
        s_low = s_min_theoretical
        s_high = s_start
        
        for _ in range(15):
            if time.time() > deadline:
                break
            s_mid = (s_low + s_high) / 2
            if _is_feasible(x0, s_mid, angle_inner, angle_outer):
                s_high = s_mid
            else:
                s_low = s_mid
        
        best_s = s_high
        if _is_feasible(x0, best_s, angle_inner, angle_outer):
            centers = x0.reshape(12, 2)
            inner_data = np.column_stack([centers, np.full(12, angle_inner)])
            outer_data = np.array([0, 0, angle_outer])
            best_result = (inner_data, outer_data, best_s)
            best_score = 1.0 / best_s

    # Try optimization to improve
    if time.time() < deadline - 60:
        s_opt = best_s if best_result else s_min_theoretical
        
        result = minimize(_objective, x0, args=(s_opt, angle_inner, angle_outer),
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
        
        if result.fun < 1e-8:
            if _is_feasible(result.x, s_opt, angle_inner, angle_outer):
                centers = result.x.reshape(12, 2)
                inner_data = np.column_stack([centers, np.full(12, angle_inner)])
                outer_data = np.array([0, 0, angle_outer])
                new_result = (inner_data, outer_data, s_opt)
                new_score = 1.0 / s_opt
                if new_score > best_score:
                    best_result = new_result
                    best_score = new_score

    if best_result is None:
        # Fallback to previous best known (C-shape)
        r2_corners = [
            axial_to_cartesian(2, 0),
            axial_to_cartesian(2, -1),
            axial_to_cartesian(1, -2),
            axial_to_cartesian(0, -2),
            axial_to_cartesian(-1, -1)
        ]
        all_centers = centers_r1 + r2_corners
        x0 = np.array(all_centers).flatten()
        angle_inner = 0.0
        angle_outer = 0.0
        s_fallback = 2 * sqrt3 + 1.0
        if _is_feasible(x0, s_fallback, angle_inner, angle_outer):
            centers = x0.reshape(12, 2)
            inner_data = np.column_stack([centers, np.full(12, angle_inner)])
            outer_data = np.array([0, 0, angle_outer])
            best_result = (inner_data, outer_data, s_fallback)
            best_score = 1.0 / s_fallback

    if best_result is None:
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
    
    return best_result

# EVOLVE-BLOCK-END
