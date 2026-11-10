import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.deg2rad(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            theta = a + np.deg2rad(60 * i)
            verts[i, 0] = cx + side * np.cos(theta)
            verts[i, 1] = cy + side * np.sin(theta)
        return verts

    def calc_min_s(centers, angles):
        all_verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        normals = []
        for k in range(6):
            theta = np.deg2rad(30 + 60 * k)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        
        s = (2.0 / np.sqrt(3)) * max_proj
        return s

    def get_overlap_penalty(centers, angles):
        penalty = 0.0
        for i in range(12):
            v_i = get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            for j in range(i + 1, 12):
                v_j = get_hex_vertices(centers[j, 0], centers[j, 1], angles[j])
                min_pen = float('inf')
                separated = False
                for poly in [v_i, v_j]:
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm_len = np.linalg.norm(normal)
                        if norm_len < 1e-6:
                            continue
                        normal = normal / norm_len
                        
                        proj_i = v_i @ normal
                        proj_j = v_j @ normal
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        if max_i < min_j or max_j < min_i:
                            separated = True
                            break
                        else:
                            pen = min(max_i, max_j) - max(min_i, min_j)
                            if pen < min_pen:
                                min_pen = pen
                    if separated:
                        break
                if not separated:
                    penalty += min_pen ** 2
        return penalty

    def objective(params):
        centers = params[0:24].reshape(12, 2)
        angles = params[24:36]
        
        s = calc_min_s(centers, angles)
        pen = get_overlap_penalty(centers, angles)
        
        return s + 100.0 * pen

    # Initial guess: 3x4 grid
    centers_init = []
    angles_init = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3)
            y = r * 1.5
            if r % 2 == 1:
                x += np.sqrt(3) / 2
            centers_init.append([x, y])
            angles_init.append(0.0)
            
    centers_init = np.array(centers_init)
    centers_init -= np.mean(centers_init, axis=0)
    
    params_init = np.concatenate([centers_init.flatten(), angles_init])
    
    best_params = params_init
    best_val = objective(params_init)
    
    # Optimize
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass

    # Random restarts
    while time.time() - start_time < deadline - 60:
        noise = np.random.normal(0, 0.5, size=params_init.shape)
        params_try = params_init + noise
        try:
            res = minimize(objective, params_try, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except Exception:
            continue

    centers = best_params[0:24].reshape(12, 2)
    angles = best_params[24:36]
    
    s_final = calc_min_s(centers, angles)
    
    inner_hex_data = np.column_stack((centers, angles))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final
