import numpy as np
import time
from scipy.optimize import minimize

N = 12

# Pre-compute constants
NORMALS = np.array([
    [np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))]
    for k in range(6)
])
COS30 = np.sqrt(3)/2

def hexagon_vertices_fast(x, y, ang_deg):
    ang = np.deg2rad(ang_deg)
    c = np.cos(ang)
    s = np.sin(ang)
    verts_local = np.array([
        [1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    R_mat = np.array([[c, -s], [s, c]])
    verts_rot = verts_local @ R_mat.T
    return verts_rot + np.array([x, y])

def min_R_for_config_fast(config):
    max_r = 0
    for i in range(N):
        x, y, ang = config[i]
        verts = hexagon_vertices_fast(x, y, ang)
        dots = verts @ NORMALS.T
        r_req = np.max(np.abs(dots)) / COS30
        if r_req > max_r:
            max_r = r_req
    return max_r

def sat_overlap_depth_fast(v1, v2):
    e1 = np.roll(v1, -1, axis=0) - v1
    e2 = np.roll(v2, -1, axis=0) - v2
    edges = np.vstack([e1, e2])
    
    min_depth = np.inf
    for edge in edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-10: continue
        normal /= norm
        
        p1 = v1 @ normal
        p2 = v2 @ normal
        
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if max1 < min2 else min1 - max2
            min_depth = min(min_depth, gap)
        else:
            overlap = min(max1, max2) - max(min1, min2)
            min_depth = min(min_depth, -overlap)
            
    return min_depth

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_side = 40.0
    best_config = None
    
    # Grid initialization
    def grid_init():
        dx = 2.0
        dy = np.sqrt(3)
        config = []
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                config.append([x, y, 0.0])
        return np.array(config)

    inits = [grid_init()]
    
    # Add random inits
    np.random.seed(42)
    for _ in range(5):
        x = np.random.uniform(-5, 5, N)
        y = np.random.uniform(-5, 5, N)
        ang = np.random.uniform(0, 60, N)
        inits.append(np.column_stack((x, y, ang)))
        
    for init_cfg in inits:
        if time.time() > deadline:
            break
            
        def objective(params):
            cfg = params.reshape(N, 3)
            r_min = min_R_for_config_fast(cfg)
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    v1 = hexagon_vertices_fast(cfg[i][0], cfg[i][1], cfg[i][2])
                    v2 = hexagon_vertices_fast(cfg[j][0], cfg[j][1], cfg[j][2])
                    d = sat_overlap_depth_fast(v1, v2)
                    if d < 0:
                        penalty += (-d)**2
                    
            return r_min + 10.0 * penalty
        
        try:
            res = minimize(objective, init_cfg.flatten(), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success:
                final_cfg = res.x.reshape(N, 3)
                # Validate
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = hexagon_vertices_fast(final_cfg[i][0], final_cfg[i][1], final_cfg[i][2])
                        v2 = hexagon_vertices_fast(final_cfg[j][0], final_cfg[j][1], final_cfg[j][2])
                        d = sat_overlap_depth_fast(v1, v2)
                        if d < 0:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side = min_R_for_config_fast(final_cfg)
                    if side < best_side:
                        best_side = side
                        best_config = final_cfg
                        
        except Exception:
            pass

    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    else:
        inner = [[float(row[0]), float(row[1]), float(row[2])] for row in best_config]
        return inner, (0.0, 0.0), float(best_side), 0.0
