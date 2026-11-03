import numpy as np
from scipy.optimize import minimize
import time

def hexagon_vertices(cx, cy, s, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle + k * np.pi / 3
        x = cx + s * np.cos(a)
        y = cy + s * np.sin(a)
        verts.append([x, y])
    return np.array(verts)

def min_enclosing_hexagon_side(points, cx, cy, angle_deg):
    angle = np.radians(angle_deg)
    normals = np.array([[np.cos(angle + np.pi/6 + k*np.pi/3), np.sin(angle + np.pi/6 + k*np.pi/3)] for k in range(6)])
    rel = points - np.array([cx, cy])
    projs = rel @ normals.T
    max_proj = np.max(projs)
    if max_proj <= 0:
        return 0.0
    return 2 * max_proj / np.sqrt(3)

def objective(params):
    n_inner = 11
    inner_params = params[:n_inner*3].reshape(n_inner, 3)
    ocx, ocy, oangle = params[n_inner*3:]
    
    all_verts = []
    for i in range(n_inner):
        cx, cy, ang = inner_params[i]
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    return min_enclosing_hexagon_side(all_verts, ocx, ocy, oangle)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial guess
    init = [[0.0, 0.0, 0.0]]
    d1 = np.sqrt(3)
    for k in range(6):
        a = k * np.pi / 3
        init.append([d1*np.cos(a), d1*np.sin(a), 0.0])
    d2 = 2*np.sqrt(3)
    for k in [1, 3, 5, 7]:
        a = k * np.pi / 6
        init.append([d2*np.cos(a), d2*np.sin(a), 0.0])
    init.append([0.0, 0.0, 0.0])
    x0 = np.array(init)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Local optimization from multiple starts
    for trial in range(10):
        if time.time() > deadline:
            break
        if trial == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(len(x0)) * 0.3
        
        res = minimize(objective, x_start, method='L-BFGS-B',
                      options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    # Final polish
    if time.time() < deadline - 5:
        res = minimize(objective, best_x, method='L-BFGS-B',
                      options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-16})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    n_inner = 11
    inner_params = best_x[:n_inner*3].reshape(n_inner, 3)
    ocx, ocy, oangle = best_x[n_inner*3:]
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in inner_params]
    
    return inner, (float(ocx), float(ocy)), float(best_val), float(oangle)
