import numpy as np
from scipy.optimize import minimize_scalar
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0 # seconds
    
    sqrt3 = np.sqrt(3)
    
    # 13-hexagon cluster centers
    centers_13 = np.array([
        [0, 0],
        [1.5, sqrt3/2],
        [0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0, -sqrt3],
        [1.5, -sqrt3/2],
        [3, sqrt3],
        [1.5, 1.5*sqrt3],
        [-1.5, 1.5*sqrt3],
        [-3, sqrt3],
        [-1.5, -1.5*sqrt3],
        [1.5, -1.5*sqrt3]
    ])
    
    # Remove one outer hexagon (index 7) to get 12
    centers = np.delete(centers_13, 7, axis=0)
    
    base_verts = np.array([
        [1, 0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1, 0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    def get_s(phi):
        if time.time() > deadline - 5:
            return 1e10 # Penalize to stop optimization
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        rot_centers = centers @ R.T
        rot_verts = base_verts @ R.T
        
        all_points = rot_centers[:, np.newaxis, :] + rot_verts[np.newaxis, :, :]
        dots = np.einsum('ijk,k->ij', all_points, normals)
        max_proj = np.max(dots, axis=0)
        R_req = np.max(max_proj)
        s_val = 2 * R_req / sqrt3
        return s_val

    # Optimize
    try:
        res = minimize_scalar(get_s, bounds=(0, np.pi/6), method='bounded')
        phi_opt = res.x
        s_opt = res.fun
    except Exception:
        phi_opt = 0.0
        s_opt = get_s(0.0)

    c, s = np.cos(phi_opt), np.sin(phi_opt)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = final_centers[:, 0]
    inner_data[:, 1] = final_centers[:, 1]
    inner_data[:, 2] = np.degrees(phi_opt)
    
    outer_data = np.array([0, 0, 0])
    
    return inner_data, outer_data, s_opt
