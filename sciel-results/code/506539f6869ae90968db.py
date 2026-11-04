import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate a tight hexagonal cluster of 12 hexagons
    # Lattice vectors for unit hexagons (side 1)
    # Center distance is sqrt(3)
    # Axial coordinates (i, j)
    # pos = i * (sqrt(3), 0) + j * (sqrt(3)/2, 3/2)
    # dist^2 = 3 * (i^2 + i*j + j^2)
    
    candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d2 = 3 * (i*i + i*j + j*j)
            candidates.append((d2, i, j))
    
    candidates.sort()
    selected = candidates[:N]
    
    centers = np.zeros((N, 2))
    for k, (d2, i, j) in enumerate(selected):
        x = i * np.sqrt(3) + j * np.sqrt(3) / 2
        y = j * 1.5
        centers[k] = [x, y]
        
    centers -= np.mean(centers, axis=0)
    
    angles = np.zeros(N)
    
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = 0.0
        
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_inner_verts(params):
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            v = (R @ base_verts.T).T + np.array([cx, cy])
            verts_list.append(v)
        return np.array(verts_list)
        
    def calc_L_from_verts(verts):
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        
        term1 = x + y / np.sqrt(3)
        term2 = 2 * y / np.sqrt(3)
        term3 = -x + y / np.sqrt(3)
        term4 = -x - y / np.sqrt(3)
        term5 = -2 * y / np.sqrt(3)
        term6 = x - y / np.sqrt(3)
        
        L_points = np.max(np.stack([term1, term2, term3, term4, term5, term6]), axis=0)
        return np.max(L_points)
        
    def get_width(params, i, dir_vec):
        ang_i = params[i*3+2]
        d_ang = np.arctan2(dir_vec[1], dir_vec[0])
        local_d = d_ang - ang_i
        diffs = local_d - base_angles
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        min_diff = np.min(np.abs(diffs))
        h = np.cos(min_diff)
        return 2.0 * h

    def overlap_penalty(params):
        penalty = 0.0
        for i in range(N):
            ci = np.array([params[i*3], params[i*3+1]])
            for j in range(i+1, N):
                cj = np.array([params[j*3], params[j*3+1]])
                diff = cj - ci
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                dir_vec = diff / dist
                w_i = get_width(params, i, dir_vec)
                w_j = get_width(params, j, -dir_vec)
                min_dist_req = (w_i + w_j) / 2.0
                if dist < min_dist_req:
                    violation = min_dist_req - dist
                    penalty += violation**2
        return penalty
        
    def objective(params):
        verts = get_inner_verts(params)
        L = calc_L_from_verts(verts)
        pen = overlap_penalty(params)
        return L + 100.0 * pen
        
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000, 'adaptive': True})
    
    best_params = res.x
    best_L = calc_L_from_verts(get_inner_verts(best_params))
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
