import numpy as np
import time

N = 11

def get_hex_vertices():
    return np.array([
        [np.sqrt(3)/2, 0.5],
        [0, 1],
        [-np.sqrt(3)/2, 0.5],
        [-np.sqrt(3)/2, -0.5],
        [0, -1],
        [np.sqrt(3)/2, -0.5]
    ])

HEX_VERTS = get_hex_vertices()
HX = HEX_VERTS[:, 0]
HY = HEX_VERTS[:, 1]

def objective(params):
    inner_params = params[:N*3]
    cx, cy, tc = params[N*3], params[N*3+1], params[N*3+2]
    
    ip = inner_params.reshape(N, 3)
    xs = ip[:, 0]
    ys = ip[:, 1]
    ts = ip[:, 2]
    
    cos_t = np.cos(ts)
    sin_t = np.sin(ts)
    
    v_rot_x = HX[:, None] * cos_t[None, :] - HY[:, None] * sin_t[None, :]
    v_rot_y = HX[:, None] * sin_t[None, :] + HY[:, None] * cos_t[None, :]
    
    v_x = v_rot_x.T + xs[None, :]
    v_y = v_rot_y.T + ys[None, :]
    
    vx = v_x.flatten()
    vy = v_y.flatten()
    
    vx_rel = vx - cx
    vy_rel = vy - cy
    
    angles = tc + np.arange(6) * (np.pi / 3)
    nx = np.cos(angles)
    ny = np.sin(angles)
    
    proj = vx_rel[:, None] * nx[None, :] + vy_rel[:, None] * ny[None, :]
    
    max_proj = np.max(np.abs(proj))
    
    S = max_proj / (np.sqrt(3) / 2)
    
    return S

def construct():
    start_time = time.time()
    deadline = 110.0
    
    init_params = np.zeros(N * 3 + 3)
    
    idx = 0
    for k in range(-1, 2):
        init_params[idx*3] = k * np.sqrt(3)
        init_params[idx*3+1] = -1.5
        init_params[idx*3+2] = 0.0
        idx += 1
        
    for k in range(-2, 3):
        init_params[idx*3] = k * np.sqrt(3)
        init_params[idx*3+1] = 0.0
        init_params[idx*3+2] = 0.0
        idx += 1
        
    for k in range(-1, 2):
        init_params[idx*3] = k * np.sqrt(3)
        init_params[idx*3+1] = 1.5
        init_params[idx*3+2] = 0.0
        idx += 1
        
    init_params[N*3] = 0.0
    init_params[N*3+1] = 0.0
    init_params[N*3+2] = 0.0
    
    best_params = init_params.copy()
    best_score = objective(best_params)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.98
    step_size = 0.5
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while T > T_min and time.time() < deadline:
        new_params = best_params + np.random.normal(0, step_size, best_params.shape)
        new_score = objective(new_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params = new_params
            no_improve_count = 0
            step_size *= 0.99
        else:
            delta = new_score - best_score
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                best_params = new_params
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 0.01:
                    break
                    
        T *= alpha
        
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except:
            pass
            
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        angle_deg = np.degrees(best_params[i*3+2])
        inner.append([x, y, angle_deg])
        
    cx = best_params[N*3]
    cy = best_params[N*3+1]
    side = best_score
    angle_deg = np.degrees(best_params[N*3+2])
    
    return inner, (cx, cy), side, angle_deg
