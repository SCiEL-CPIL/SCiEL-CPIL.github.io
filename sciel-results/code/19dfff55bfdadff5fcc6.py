import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + np.radians(60 * i)),
                      cy + side * np.sin(a + np.radians(60 * i))] for i in range(6)])

def polygons_overlap(v1, v2):
    for verts in (v1, v2):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            min1, max1 = (v1 @ normal).min(), (v1 @ normal).max()
            min2, max2 = (v2 @ normal).min(), (v2 @ normal).max()
            if max1 < min2 or max2 < min1:
                return False
    return True

def point_in_polygon(px, py, verts):
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        if cross < -1e-9:
            return False
    return True

def polygon_contained(v_inner, v_outer):
    for pt in v_inner:
        if not point_in_polygon(pt[0], pt[1], v_outer):
            return False
    return True

def min_penetration(v1, v2):
    min_pen = np.inf
    for verts in (v1, v2):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            min1, max1 = (v1 @ normal).min(), (v1 @ normal).max()
            min2, max2 = (v2 @ normal).min(), (v2 @ normal).max()
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                min_pen = min(min_pen, overlap)
    if min_pen == np.inf:
        return 0.0
    return min_pen

def max_containment_violation(v_inner, v_outer):
    max_viol = 0.0
    for pt in v_inner:
        for i in range(6):
            a = v_outer[i]
            b = outer_verts[(i + 1) % 6] if False else v_outer[(i + 1) % 6]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            dist = np.dot(pt - a, normal)
            if dist > max_viol:
                max_viol = dist
    return max_viol

def objective(params):
    inner_xy = params[:2 * N].reshape(N, 2)
    outer_cx, outer_cy, outer_side, outer_angle = params[2 * N:2 * N + 4]
    
    if outer_side < 1.0:
        return 1e6
    
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    penalty = 0.0
    inner_verts_list = [hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy]
    
    for i in range(N):
        viol = max_containment_violation(inner_verts_list[i], outer_verts)
        if viol > 0:
            penalty += viol ** 2
    
    for i in range(N):
        for j in range(i + 1, N):
            pen = min_penetration(inner_verts_list[i], inner_verts_list[j])
            if pen > 0:
                penalty += pen ** 2
    
    return outer_side + 500.0 * penalty

def construct():
    start_time = time.time()
    sqrt3 = np.sqrt(3)
    
    initial_inner = [
        [-sqrt3, 0.0], [0.0, 0.0], [sqrt3, 0.0],
        [-1.5*sqrt3, 1.5], [-0.5*sqrt3, 1.5], [0.5*sqrt3, 1.5], [1.5*sqrt3, 1.5],
        [-sqrt3, 3.0], [0.0, 3.0], [sqrt3, 3.0],
        [0.0, 4.5],
    ]
    
    best_params = None
    best_score = np.inf
    
    for oa in [0, 30, 60, 90, 120, 150]:
        if time.time() - start_time > DEADLINE:
            break
        
        params0 = np.array([
            *[p for row in initial_inner for p in row],
            0.0, 2.25, 3.5, float(oa)
        ])
        
        result = minimize(objective, params0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    
    if best_params is None:
        inner = [[float(x), float(y), 0.0] for x, y in initial_inner]
        return inner, (0.0, 2.25), 10.0, 0.0
    
    inner_xy = best_params[:2 * N].reshape(N, 2)
    outer_cx, outer_cy, outer_side, outer_angle = best_params[2 * N:2 * N + 4]
    
    # Validate
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    inner_verts_list = [hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy]
    
    valid = True
    for i in range(N):
        if not polygon_contained(inner_verts_list[i], outer_verts):
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        for _ in range(200):
            outer_side *= 1.01
            outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
            valid = True
            for i in range(N):
                if not polygon_contained(inner_verts_list[i], outer_verts):
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                break
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_xy]
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)
