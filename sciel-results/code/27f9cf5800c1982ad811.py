import numpy as np
from scipy.optimize import minimize

N = 12

def get_hex_vertices(cx, cy, angle_deg):
    angle = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def calculate_min_side(params, outer_rot_rad):
    all_vertices = []
    for i in range(N):
        cx = params[i*3]
        cy = params[i*3+1]
        angle = params[i*3+2]
        verts = get_hex_vertices(cx, cy, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    cos_r = np.cos(-outer_rot_rad)
    sin_r = np.sin(-outer_rot_rad)
    x_rot = all_vertices[:, 0] * cos_r - all_vertices[:, 1] * sin_r
    y_rot = all_vertices[:, 0] * sin_r + all_vertices[:, 1] * cos_r
    
    normal_angles = np.arange(6) * np.pi / 3 + np.pi / 6
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    max_proj = 0
    for k in range(6):
        dots = x_rot * nx[k] + y_rot * ny[k]
        max_proj = max(max_proj, np.max(dots))
        
    side = (2.0 / np.sqrt(3.0)) * max_proj
    return side

def objective(params):
    outer_rot_rad = params[-1]
    inner_params = params[:-1]
    return calculate_min_side(inner_params, outer_rot_rad)

def construct():
    sqrt3 = np.sqrt(3)
    centers = [
        (0, 0),
        (sqrt3, 0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5)
    ]
    
    best_side = float('inf')
    best_inner = None
    best_outer_rot = 0.0
    
    # Try a few random restarts
    for restart in range(5):
        init_params = []
        for i in range(N):
            if i < 7:
                cx, cy = centers[i]
            else:
                angle = (i-7) * 60
                rad = np.deg2rad(angle)
                cx = 2 * sqrt3 * np.cos(rad)
                cy = 2 * sqrt3 * np.sin(rad)
            
            # Add random jitter
            cx += np.random.uniform(-0.5, 0.5)
            cy += np.random.uniform(-0.5, 0.5)
            angle_deg = np.random.uniform(-10, 10)
            init_params.extend([cx, cy, angle_deg])
        
        init_params.append(np.random.uniform(0, 60))
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                inner_params = best_params[:-1]
                best_outer_rot = np.rad2deg(best_params[-1])
                
                inner = []
                for i in range(N):
                    cx = inner_params[i*3]
                    cy = inner_params[i*3+1]
                    angle = inner_params[i*3+2]
                    inner.append([cx, cy, angle])
                best_inner = inner
        except Exception:
            pass
            
    if best_inner is None:
        # Fallback to initial guess
        inner = []
        for i in range(N):
            if i < 7:
                cx, cy = centers[i]
            else:
                angle = (i-7) * 60
                rad = np.deg2rad(angle)
                cx = 2 * sqrt3 * np.cos(rad)
                cy = 2 * sqrt3 * np.sin(rad)
            inner.append([cx, cy, 0.0])
        best_side = calculate_min_side(np.array([p for h in inner for p in h]), 0.0)
        best_outer_rot = 0.0

    return best_inner, (0.0, 0.0), best_side, best_outer_rot
