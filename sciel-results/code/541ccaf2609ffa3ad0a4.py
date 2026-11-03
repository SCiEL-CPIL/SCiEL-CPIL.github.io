import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    
    # Initial configuration
    coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 2)
    ]
    
    def hex_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    initial_positions = np.array([hex_to_cart(q, r) for q, r in coords])
    centroid = np.mean(initial_positions, axis=0)
    initial_positions -= centroid
    
    x0 = initial_positions.flatten()
    cx0, cy0 = 0.0, 0.0
    phi0 = 0.0
    x_init = np.concatenate([x0, [cx0, cy0, phi0]])
    
    def get_vertices(positions):
        verts = []
        for i in range(12):
            cx, cy = positions[i]
            angles = np.deg2rad(np.arange(0, 360, 60))
            vx = cx + np.cos(angles)
            vy = cy + np.sin(angles)
            verts.append(np.column_stack((vx, vy)))
        return np.array(verts).reshape(-1, 2)

    def calc_min_side(positions, cx, cy, phi):
        verts = get_vertices(positions)
        # Translate vertices relative to container center
        verts_rel = verts - np.array([cx, cy])
        
        base_angles = np.deg2rad([90, 150, 210])
        max_proj = 0
        for a in base_angles:
            a_rad = a + phi
            n = np.array([np.cos(a_rad), np.sin(a_rad)])
            projs = np.abs(verts_rel @ n)
            max_proj = max(max_proj, np.max(projs))
            
        return (2.0 / np.sqrt(3.0)) * max_proj

    def objective(params):
        if time.time() - start_time > 120:
            return 1e9
        
        positions = params[:24].reshape(12, 2)
        cx = params[24]
        cy = params[25]
        phi = params[26]
        
        # Overlap penalty
        min_dist = np.sqrt(3)
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        side = calc_min_side(positions, cx, cy, phi)
        return side + 100.0 * penalty

    best_x = x_init
    best_val = objective(x_init)
    
    try:
        res = minimize(objective, x_init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if time.time() - start_time < 140:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass

    best_positions = best_x[:24].reshape(12, 2)
    best_cx = best_x[24]
    best_cy = best_x[25]
    best_phi = best_x[26]
    
    best_side = calc_min_side(best_positions, best_cx, best_cy, best_phi)
    
    inner = []
    for i in range(12):
        x, y = best_positions[i]
        inner.append([x, y, 0.0])
        
    return inner, [best_cx, best_cy], best_side, best_phi

# EVOLVE-BLOCK-END
