import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def get_hex_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals_angles = angle_rad + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    return np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))

def calculate_penalty(x, y, angles, S_outer):
    penalty = 0.0
    d_outer = S_outer * np.sqrt(3) / 2
    normals_outer = get_hex_normals(0.0)
    
    inner_verts = []
    for i in range(N):
        v = get_hex_vertices(x[i], y[i], 1.0, angles[i])
        inner_verts.append(v)
        
    for i in range(N):
        v = inner_verts[i]
        projections = v @ normals_outer.T
        max_proj = np.max(projections)
        if max_proj > d_outer:
            penalty += (max_proj - d_outer) ** 2
            
    for i in range(N):
        for j in range(i+1, N):
            v_i = inner_verts[i]
            v_j = inner_verts[j]
            
            normals_i = get_hex_normals(angles[i])
            normals_j = get_hex_normals(angles[j])
            all_normals = np.vstack((normals_i, normals_j))
            
            max_gap = -np.inf
            for k in range(12):
                n = all_normals[k]
                proj_i = v_i @ n
                proj_j = v_j @ n
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                gap = min(max_i, max_j) - max(min_i, min_j)
                if gap > max_gap:
                    max_gap = gap
                    
            if max_gap < 0:
                penalty += (-max_gap) ** 2
                
    return penalty

def objective(params):
    x = params[:12]
    y = params[12:24]
    angles = params[24:36]
    S_outer = params[36]
    
    if S_outer < 1.0:
        return 1000.0 + calculate_penalty(x, y, angles, S_outer)
        
    return S_outer + 10.0 * calculate_penalty(x, y, angles, S_outer)

def generate_layout(type_id):
    if type_id == 0:
        q_vals = np.array([0, 1, 2, 3])
        r_vals = np.array([0, 1, 2])
    elif type_id == 1:
        q_vals = np.array([0, 1, 2, 3, 4, 5])
        r_vals = np.array([0, 1])
    else:
        coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        next_ring = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-3)]
        coords.extend(next_ring[:5])
        q_vals = np.array([c[0] for c in coords])
        r_vals = np.array([c[1] for c in coords])
        
    x = []
    y = []
    for r in r_vals:
        for q in q_vals:
            x.append(np.sqrt(3) * (q + 0.5 * r))
            y.append(1.5 * r)
            
    x = np.array(x)
    y = np.array(y)
    x -= np.mean(x)
    y -= np.mean(y)
    
    max_r = np.max(np.sqrt(x**2 + y**2))
    S = max_r + 2.0
    
    angles = np.zeros(12)
    return np.concatenate((x, y, angles, [S]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = np.inf
    best_params = None
    
    bounds = [(-20, 20)] * 24 + [(0, 360)] * 12 + [(1.0, 20.0)]
    
    for layout_type in range(3):
        base_params = generate_layout(layout_type)
        
        for restart in range(10):
            if time.time() > deadline:
                break
                
            noise = np.random.normal(0, 1.0, size=36)
            noise[36] = 0
            current_params = base_params + noise
            
            try:
                res = minimize(objective, current_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
                if res.success:
                    x = res.x[:12]
                    y = res.x[12:24]
                    angles = res.x[24:36]
                    S_val = res.x[36]
                    pen = calculate_penalty(x, y, angles, S_val)
                    if pen < 1e-5:
                        if S_val < best_score:
                            best_score = S_val
                            best_params = res.x
            except Exception:
                pass

    if best_params is None:
        # Fallback
        best_params = generate_layout(0)
        best_score = best_params[36]

    x = best_params[:12]
    y = best_params[12:24]
    angles = best_params[24:36]
    S_outer = best_params[36]
    
    inner = [[x[i], y[i], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), S_outer, 0.0
