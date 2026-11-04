import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    angles = np.deg2rad(angle_deg + np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def point_in_convex_polygon(p, poly):
    signs = []
    for i in range(6):
        p1 = poly[i]
        p2 = poly[(i+1)%6]
        v = p2 - p1
        w = p - p1
        cross = v[0]*w[1] - v[1]*w[0]
        signs.append(cross)
    if any(s > 1e-9 for s in signs) and any(s < -1e-9 for s in signs):
        return False
    return True

def check_overlap(hex1_verts, hex2_verts):
    for v in hex1_verts:
        if point_in_convex_polygon(v, hex2_verts):
            return True
    for v in hex2_verts:
        if point_in_convex_polygon(v, hex1_verts):
            return True
    return False

def minimal_enclosing_hex_side(points, center):
    def calc_S(phi):
        widths = []
        for k in range(3):
            theta = phi + k * np.pi / 3
            u = np.array([np.cos(theta), np.sin(theta)])
            proj = (points - center) @ u
            w = np.max(proj) - np.min(proj)
            widths.append(w)
        return np.max(widths) / np.sqrt(3)
        
    res = minimize_scalar(calc_S, bounds=(0, np.pi/3), method='bounded')
    return res.fun, res.x

def objective(params):
    positions = np.zeros((11, 3))
    positions[0] = [0.0, 0.0, 0.0]
    positions[1:] = np.array(params).reshape(10, 3)
    
    all_verts = []
    hex_verts = []
    for i in range(11):
        cx, cy, ang = positions[i]
        verts = get_hex_vertices(cx, cy, ang)
        hex_verts.append(verts)
        all_verts.append(verts)
        
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(hex_verts[i], hex_verts[j]):
                return 1000.0
                
    all_verts_flat = np.vstack(all_verts)
    center = np.mean(all_verts_flat, axis=0)
    S, _ = minimal_enclosing_hex_side(all_verts_flat, center)
    return S

def construct():
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i*v1 + j*v2
            candidates.append(p)
            
    candidates = sorted(candidates, key=lambda p: np.dot(p, p))
    init_positions = np.array(candidates[:11])
    
    init_params = init_positions[1:].flatten()
    
    start_time = time.time()
    timeout = 140.0
    
    best_params = init_params
    best_score = objective(best_params)
    
    def time_checking_callback(xk):
        nonlocal best_params, best_score
        score = objective(xk)
        if score < best_score:
            best_score = score
            best_params = xk.copy()
        if time.time() - start_time > timeout:
            raise TimeoutError("Time limit reached")
            
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6},
                       callback=time_checking_callback)
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except TimeoutError:
        pass
    except Exception:
        pass
        
    final_positions = np.zeros((11, 3))
    final_positions[0] = [0.0, 0.0, 0.0]
    final_positions[1:] = best_params.reshape(10, 3)
    
    all_verts = []
    for i in range(11):
        cx, cy, ang = final_positions[i]
        verts = get_hex_vertices(cx, cy, ang)
        all_verts.append(verts)
        
    all_verts_flat = np.vstack(all_verts)
    center = np.mean(all_verts_flat, axis=0)
    S, phi_opt = minimal_enclosing_hex_side(all_verts_flat, center)
    
    inner = [[float(x), float(y), float(np.degrees(t))] for x, y, t in final_positions]
    outer_center = [float(center[0]), float(center[1])]
    outer_side = float(S)
    outer_angle = float(np.degrees(phi_opt) - 30.0)
    
    return inner, outer_center, outer_side, outer_angle
