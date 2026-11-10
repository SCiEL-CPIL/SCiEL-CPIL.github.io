import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    n = 12
    sqrt3 = np.sqrt(3)
    
    # Normals for flat-top hexagon (angle 0)
    # n_k = (cos(60k), sin(60k))
    angles = np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_hex_vertices(cx, cy, d, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(60 * np.arange(6))
        xs = cx + d * np.cos(angles)
        ys = cy + d * np.sin(angles)
        return np.column_stack((xs, ys))

    def objective(params):
        centers = params.reshape(12, 2)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                diff = centers[j] - centers[i]
                # Project diff onto normals
                projs = normals @ diff
                # Check if |proj| > sqrt(3)
                violations = np.abs(projs) - sqrt3
                max_violation = np.max(violations)
                if max_violation > 0:
                    penalty += max_violation ** 2
        
        # Calculate minimal enclosing hexagon side s
        verts_list = []
        for i in range(12):
            v = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        
        best_s = float('inf')
        # Sample outer angles
        for alpha_deg in np.linspace(0, 60, 36):
            alpha_rad = np.deg2rad(alpha_deg)
            out_angles = alpha_rad + np.deg2rad(60 * np.arange(6))
            out_normals = np.column_stack((np.cos(out_angles), np.sin(out_angles)))
            
            proj = all_verts @ out_normals.T
            max_abs = np.max(np.abs(proj))
            s = (2.0 / sqrt3) * max_abs
            if s < best_s:
                best_s = s
                
        return best_s + 100.0 * penalty

    # Initial guess
    x0 = []
    y0 = []
    for x in [-3, -1, 1, 3]:
        x0.append(x); y0.append(0)
    for x in [-2, 0, 2, 4]:
        x0.append(x); y0.append(sqrt3)
    for x in [-3, -1, 1, 3]:
        x0.append(x); y0.append(2*sqrt3)
        
    x0 = np.array(x0); y0 = np.array(y0)
    cx_mean = np.mean(x0); cy_mean = np.mean(y0)
    x0 = x0 - cx_mean; y0 = y0 - cy_mean
    
    initial_params = np.column_stack((x0, y0)).flatten()
    
    result = minimize(objective, initial_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = result.x
    centers = best_params.reshape(12, 2)
    
    # Final s calculation
    verts_list = []
    for i in range(12):
        v = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        verts_list.append(v)
    all_verts = np.vstack(verts_list)
    
    best_s = float('inf')
    best_alpha = 0
    for alpha_deg in np.linspace(0, 60, 360):
        alpha_rad = np.deg2rad(alpha_deg)
        out_angles = alpha_rad + np.deg2rad(60 * np.arange(6))
        out_normals = np.column_stack((np.cos(out_angles), np.sin(out_angles)))
        proj = all_verts @ out_normals.T
        max_abs = np.max(np.abs(proj))
        s = (2.0 / sqrt3) * max_abs
        if s < best_s:
            best_s = s
            best_alpha = alpha_deg

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, best_alpha])
    
    return inner_hex_data, outer_hex_data, best_s
