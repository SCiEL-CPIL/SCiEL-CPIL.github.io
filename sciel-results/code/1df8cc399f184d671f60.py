import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(angle_rad + i * np.pi / 3),
                      cy + side * np.sin(angle_rad + i * np.pi / 3)] for i in range(6)])

def _get_all_inner_vertices(inner):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for (x, y, angle_deg) in inner:
        all_verts.append(_hex_vertices(x, y, 1.0, angle_deg))
    return np.vstack(all_verts)

def _compute_min_side(all_verts, cx, cy, theta_rad):
    """Compute minimum side length of outer hexagon."""
    normals = np.array([[np.cos(theta_rad + k * np.pi / 3),
                         np.sin(theta_rad + k * np.pi / 3)] for k in range(6)])
    rel = all_verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    return float(np.max(max_proj) / (np.sqrt(3) / 2))

def _objective_outer(params, all_verts):
    cx, cy, theta = params
    return _compute_min_side(all_verts, cx, cy, theta)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    d = np.sqrt(3)
    
    # Arrangement 1: 4-3-3-2
    pos1 = []
    for i in range(4):
        pos1.append([(i - 1.5) * d, 0.0, 0.0])
    for i in range(3):
        pos1.append([(i - 1.0) * d, 1.5, 0.0])
    for i in range(3):
        pos1.append([(i - 1.0) * d, 3.0, 0.0])
    for i in range(2):
        pos1.append([(i - 0.5) * d, 4.5, 0.0])
    
    # Arrangement 2: 3-4-3-2 (shifted)
    pos2 = []
    for i in range(3):
        pos2.append([(i - 1.0) * d, 0.0, 0.0])
    for i in range(4):
        pos2.append([(i - 1.5) * d, 1.5, 0.0])
    for i in range(3):
        pos2.append([(i - 1.0) * d, 3.0, 0.0])
    for i in range(2):
        pos2.append([(i - 0.5) * d, 4.5, 0.0])
    
    # Arrangement 3: 2-3-4-3
    pos3 = []
    for i in range(2):
        pos3.append([(i - 0.5) * d, 0.0, 0.0])
    for i in range(3):
        pos3.append([(i - 1.0) * d, 1.5, 0.0])
    for i in range(4):
        pos3.append([(i - 1.5) * d, 3.0, 0.0])
    for i in range(3):
        pos3.append([(i - 1.0) * d, 4.5, 0.0])
    
    best_S = float('inf')
    best_inner = pos1
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for positions in [pos1, pos2, pos3]:
        if time.time() > deadline - 20:
            break
        
        inner = [list(p) for p in positions]
        all_verts = _get_all_inner_vertices(inner)
        
        cx0 = float(np.mean(all_verts[:, 0]))
        cy0 = float(np.mean(all_verts[:, 1]))
        
        for theta0_deg in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline - 20:
                break
            theta0_rad = np.radians(theta0_deg)
            
            for cx_try in [cx0]:
                for cy_try in [cy0]:
                    if time.time() > deadline - 20:
                        break
                    x0 = np.array([cx_try, cy_try, theta0_rad])
                    try:
                        res = minimize(_objective_outer, x0, args=(all_verts,),
                                      method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                        S_val = res.fun
                        if S_val < best_S:
                            best_S = S_val
                            best_inner = inner
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_angle = float(np.degrees(res.x[2]))
                    except:
                        pass
    
    if best_S == float('inf'):
        # Fallback
        inner = [list(p) for p in pos1]
        all_verts = _get_all_inner_vertices(inner)
        cx0 = float(np.mean(all_verts[:, 0]))
        cy0 = float(np.mean(all_verts[:, 1]))
        best_S = _compute_min_side(all_verts, cx0, cy0, 0.0)
        best_center = (cx0, cy0)
        best_angle = 0.0
    
    return best_inner, best_center, float(best_S), float(best_angle)
