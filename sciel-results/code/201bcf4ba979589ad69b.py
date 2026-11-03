import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    d2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    r = np.array([p1[0] - p3[0], p1[1] - p3[1]])
    
    a = np.dot(d1, d1)
    e = np.dot(d2, e) # Wait, e is defined as dot(d2, d2) in previous thought, but here I wrote dot(d2, e) which is wrong.
    # Correct:
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    EPS = 1e-9
    if a <= EPS and e <= EPS:
        return np.linalg.norm(r)
    if a <= EPS:
        t1 = 0.0
        t2 = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= EPS:
            t2 = 0.0
            t1 = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom > EPS:
                t1 = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                t1 = 0.0
            t2 = (b * t1 + f) / e
            if t2 < 0.0:
                t2 = 0.0
                t1 = np.clip(-c / a, 0.0, 1.0)
            elif t2 > 1.0:
                t2 = 1.0
                t1 = np.clip((b - c) / a, 0.0, 1.0)
    
    c1 = p1 + t1 * d1
    c2 = p3 + t2 * d2
    return np.linalg.norm(c1 - c2)

def _poly_dist(v1, v2):
    min_d = float('inf')
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_d:
                min_d = d
    return min_d

def _hex_support(angle_rad, side=1.0):
    a = angle_rad % (np.pi / 3.0)
    apothem = side * np.sqrt(3) / 2.0
    return apothem / np.cos(a)

def _get_edges(v):
    return [(v[i], v[(i+1)%6]) for i in range(6)]

def _point_in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def construct():
    start_time = time.time()
    deadline = 110.0
    
    def objective(params):
        L = params[0]
        theta_out = params[1]
        centers = params[2:2 + 2 * N].reshape(N, 2)
        angles_in = params[2 + 2 * N:2 + 3 * N]
        
        r_out = L * np.sqrt(3) / 2.0
        out_angles = theta_out + np.arange(6) * np.pi / 3.0
        out_normals = np.column_stack((np.cos(out_angles), np.sin(out_angles)))
        
        pen_cont = 0.0
        for i in range(N):
            c = centers[i]
            a_in = np.deg2rad(angles_in[i])
            for k in range(6):
                n = out_normals[k]
                n_angle = np.arctan2(n[1], n[0])
                rel_angle = n_angle - a_in
                h = _hex_support(rel_angle)
                dist = np.dot(c, n) + h
                if dist > r_out:
                    pen_cont += (dist - r_out) ** 2
        
        pen_ov = 0.0
        verts = []
        for i in range(N):
            v = _hex_vertices(centers[i, 0], centers[i, 1], angles_in[i])
            verts.append(v)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = _poly_dist(verts[i], verts[j])
                if d < 1e-5:
                    overlap_score = 0.0
                    edges_j = _get_edges(verts[j])
                    edges_i = _get_edges(verts[i])
                    for v in verts[i]:
                        if _point_in_poly(v, verts[j]):
                            dist_to_edge = min(_seg_seg_dist(v, v, e[0], e[1]) for e in edges_j)
                            overlap_score += dist_to_edge ** 2
                    for v in verts[j]:
                        if _point_in_poly(v, verts[i]):
                            dist_to_edge = min(_seg_seg_dist(v, v, e[0], e[1]) for e in edges_i)
                            overlap_score += dist_to_edge ** 2
                    if overlap_score > 0:
                        pen_ov += overlap_score
        
        return L + 100.0 * pen_cont + 100.0 * pen_ov

    # Initial guess
    centers_init = []
    angles_init = []
    for row in range(4):
        for col in range(3):
            x = col * 1.5 + (row % 2) * 0.75
            y = row * np.sqrt(3) / 2.0
            centers_init.append([x, y])
            angles_init.append(0.0)
            
    cx = np.mean([c[0] for c in centers_init])
    cy = np.mean([c[1] for c in centers_init])
    for i in range(N):
        centers_init[i][0] -= cx
        centers_init[i][1] -= cy
        
    params_init = []
    params_init.append(4.0)
    params_init.append(0.0)
    for i in range(N):
        params_init.append(centers_init[i][0])
        params_init.append(centers_init[i][1])
    for i in range(N):
        params_init.append(angles_init[i])
        
    params_init = np.array(params_init)
    
    best_score = float('inf')
    best_params = params_init
    
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.success:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass
        
    L = best_params[0]
    theta_out = best_params[1]
    centers = best_params[2:2 + 2 * N].reshape(N, 2)
    angles_in = best_params[2 + 2 * N:2 + 3 * N]
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(angles_in[i])])
        
    center = (0.0, 0.0)
    side = L
    angle = np.rad2deg(theta_out)
    
    return inner, center, side, angle
