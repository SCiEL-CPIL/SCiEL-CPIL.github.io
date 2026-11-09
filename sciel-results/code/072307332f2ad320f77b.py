import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = time.time() + 600 - 30

    def get_vertices(cx, cy, angle_deg, side=1.0):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def get_normals(angle_deg):
        return np.deg2rad(angle_deg + 30 + np.arange(6) * 60)

    def min_s_for_config(pos, outer_rot):
        normals = np.column_stack([np.cos(get_normals(outer_rot)), np.sin(get_normals(outer_rot))])
        max_proj = 0
        for i in range(12):
            verts = get_vertices(pos[i,0], pos[i,1], 0)
            projs = verts @ normals.T
            abs_projs = np.abs(projs)
            max_proj = max(max_proj, np.max(abs_projs))
        return (2.0 / np.sqrt(3)) * max_proj

    def overlap_penalty(pos):
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d) ** 2
        return penalty

    def objective(vars):
        if time.time() > deadline:
            return 1e10
        pos = vars[:24].reshape(12, 2)
        outer_rot = vars[24]
        s = min_s_for_config(pos, outer_rot)
        pen = overlap_penalty(pos)
        return s + 100 * pen

    best_s = 100.0
    best_pos = None
    best_rot = 0.0

    # Initial guess: 7-hex flower + 5
    centers = [[0, 0]]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    
    for _ in range(5):
        angle = np.deg2rad(np.random.uniform(0, 360))
        r = 3.0 + np.random.uniform(-0.5, 0.5)
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        
    init_pos = np.array(centers)
    
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess
        pos = init_pos + np.random.normal(0, 0.5, (12, 2))
        rot = np.random.uniform(0, 60)
        
        init_vars = np.append(pos.flatten(), rot)
        
        try:
            res = minimize(objective, init_vars, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if time.time() > deadline:
                break
            opt_vars = res.x
            opt_pos = opt_vars[:24].reshape(12, 2)
            opt_rot = opt_vars[24]
            opt_s = min_s_for_config(opt_pos, opt_rot)
            if overlap_penalty(opt_pos) < 1e-6:
                if opt_s < best_s:
                    best_s = opt_s
                    best_pos = opt_pos
                    best_rot = opt_rot
        except Exception:
            pass

    if best_pos is None:
        # Fallback to seed-like structure
        best_pos = np.array(centers)
        best_s = min_s_for_config(best_pos, 0)
        best_rot = 0

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [best_pos[i,0], best_pos[i,1], 0]
        
    outer_hex_data = np.array([0, 0, np.rad2deg(best_rot)])
    
    return inner_hex_data, outer_hex_data, best_s
