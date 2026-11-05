import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _min_enclosing_hex_side_fast(centers, angles, outer_angle_deg, outer_center):
    n = len(centers)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = np.deg2rad(angles[i])
        thetas = a + np.arange(6) * np.pi / 3
        all_verts[i*6:(i+1)*6, 0] = centers[i][0] + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = centers[i][1] + np.sin(thetas)
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = all_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _objective_outer(vars, centers, angles):
    outer_cx, outer_cy, outer_angle = vars
    outer_center = np.array([outer_cx, outer_cy])
    return _min_enclosing_hex_side_fast(centers, angles, outer_angle, outer_center)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    centers_list = []
    angles_list = []
    
    # 1. Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)
    
    # 2. Inner ring: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    # 3. Outer ring: 4 hexagons
    # Placed at radius 3*sqrt(3) to touch the inner ring
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3 * np.sqrt(3)
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    
    # Initial guess for outer hexagon
    x0 = np.array([0.0, 0.0, 30.0])
    
    best_S = float('inf')
    best_vars = x0
    
    # Optimize outer hexagon position and angle
    try:
        res = minimize(_objective_outer, x0, args=(centers, angles), method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_S:
            best_S = res.fun
            best_vars = res.x
    except Exception:
        pass
        
    # Final result
    outer_cx, outer_cy, outer_angle = best_vars
    outer_center = (outer_cx, outer_cy)
    
    inner = []
    for i in range(len(angles)):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    # Recompute side precisely
    side = _min_enclosing_hex_side_fast(centers, angles, outer_angle, np.array([outer_cx, outer_cy]))
    
    return inner, outer_center, side, outer_angle
