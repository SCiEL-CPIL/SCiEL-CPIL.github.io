import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def get_all_vertices(inner):
    all_verts = []
    for row in inner:
        cx, cy, ang = row[0], row[1], row[2]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def min_outer_side(verts, cx, cy, angle_deg):
    angle = np.radians(angle_deg)
    normals = np.array([[np.cos(angle + k*np.pi/3), np.sin(angle + k*np.pi/3)] for k in range(6)])
    diffs = verts - np.array([cx, cy])
    dots = diffs @ normals.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    return (2.0 / np.sqrt(3.0)) * global_max

def overlap_penalty(inner):
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0:
                penalty += (2.0 - dist) ** 2
    return penalty

def objective(params):
    n_inner = 3 * N
    inner_params = params[:n_inner]
    outer_params = params[n_inner:]
    
    inner = []
    for i in range(N):
        inner.append([inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]])
    
    verts = get_all_vertices(inner)
    outer_side = min_outer_side(verts, outer_params[0], outer_params[1], outer_params[2])
    pen = overlap_penalty(inner)
    return outer_side + 100.0 * pen

def make_initial_config(variant=0):
    sqrt3 = np.sqrt(3)
    inner = []
    if variant == 0:
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 + (row % 2) * sqrt3 / 2
                y = row * 1.5
                inner.append([x, y, 0.0])
    elif variant == 1:
        for row in range(4):
            for col in range(3):
                x = col * sqrt3 + (row % 2) * sqrt3 / 2
                y = row * 1.5
                inner.append([x, y, 0.0])
    elif variant == 2:
        for i in range(12):
            inner.append([i * sqrt3, 0.0, 0.0])
    elif variant == 3:
        for row in range(2):
            for col in range(6):
                x = col * sqrt3 + (row % 2) * sqrt3 / 2
                y = row * 1.5
                inner.append([x, y, 0.0])
    else:
        for i in range(12):
            r = 2.0 * (i % 4)
            theta = 60.0 * (i // 4) + 30.0 * (i % 2)
            x = r * np.cos(np.radians(theta))
            y = r * np.sin(np.radians(theta))
            inner.append([x, y, 0.0])
    
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx_mean
        inner[i][1] -= cy_mean
    
    params = []
    for p in inner:
        params.extend(p)
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def construct():
    start_time = time.time()
    best_params = None
    best_score = np.inf
    
    for variant in range(5):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        params_init = make_initial_config(variant)
        
        def obj_func(x):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                return best_score
            return objective(x)
        
        try:
            options = {'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            result = minimize(obj_func, params_init, method='Nelder-Mead', options=options)
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if best_params is not None and elapsed < DEADLINE - 15:
        def obj_func(x):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                return best_score
            return objective(x)
        
        try:
            options2 = {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            result2 = minimize(obj_func, best_params, method='Nelder-Mead', options=options2)
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    n_inner = 3 * N
    inner_params = best_params[:n_inner]
    outer_params = best_params[n_inner:]
    
    inner = []
    for i in range(N):
        inner.append([float(inner_params[3*i]), float(inner_params[3*i+1]), float(inner_params[3*i+2])])
    
    cx = float(outer_params[0])
    cy = float(outer_params[1])
    angle = float(outer_params[2])
    side = float(best_score)
    
    return inner, (cx, cy), side, angle
