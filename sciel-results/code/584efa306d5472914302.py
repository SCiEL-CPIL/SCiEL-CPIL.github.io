import numpy as np
from scipy.optimize import minimize
import time

N = 11

def enclosing_hex_side(pts, center, angle_deg=0):
    rel = pts - np.array(center)
    rad = np.deg2rad(angle_deg)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rad
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    proj = rel @ np.column_stack((cos_a, sin_a))
    max_proj = np.max(proj)
    return max_proj / (np.sqrt(3) / 2)

def make_configs():
    configs = []
    sqrt3 = np.sqrt(3)
    
    def make_hex_cluster(rows):
        pts = []
        for i, (count, ox) in enumerate(rows):
            y = i * 1.5
            for j in range(count):
                x = ox + j * sqrt3
                pts.append((x, y))
        return pts
    
    # Various row arrangements (bottom to top)
    configs.append(make_hex_cluster([(6, 0), (5, sqrt3/2)]))
    configs.append(make_hex_cluster([(3, -sqrt3), (5, -sqrt3/2), (3, -sqrt3)]))
    configs.append(make_hex_cluster([(2, -sqrt3/2), (4, -3*sqrt3/2), (3, -sqrt3), (2, -sqrt3/2)]))
    configs.append(make_hex_cluster([(3, -sqrt3), (4, -3*sqrt3/2), (3, -sqrt3), (1, 0)]))
    configs.append(make_hex_cluster([(1, 0), (3, -sqrt3), (4, -3*sqrt3/2), (3, -sqrt3)]))
    configs.append(make_hex_cluster([(2, -sqrt3/2), (3, -sqrt3), (4, -3*sqrt3/2), (2, -sqrt3/2)]))
    configs.append(make_hex_cluster([(3, -sqrt3), (3, -sqrt3), (3, -sqrt3), (2, -sqrt3/2)]))
    configs.append(make_hex_cluster([(4, -3*sqrt3/2), (4, -3*sqrt3/2), (3, -sqrt3)]))
    configs.append(make_hex_cluster([(3, -sqrt3), (4, -3*sqrt3/2), (4, -3*sqrt3/2)]))
    configs.append(make_hex_cluster([(5, -2*sqrt3), (6, -sqrt3/2)]))
    configs.append(make_hex_cluster([(4, -3*sqrt3/2), (5, -sqrt3/2), (2, -sqrt3/2)]))
    configs.append(make_hex_cluster([(2, -sqrt3/2), (5, -sqrt3/2), (4, -3*sqrt3/2)]))
    configs.append(make_hex_cluster([(1, 0), (2, -sqrt3/2), (4, -3*sqrt3/2), (3, -sqrt3), (1, 0)]))
    configs.append(make_hex_cluster([(1, 0), (3, -sqrt3), (3, -sqrt3), (3, -sqrt3), (1, 0)]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    configs = make_configs()
    
    best_side = float('inf')
    best_config = None
    best_center = None
    best_angle = 0.0
    
    for config in configs:
        if time.time() > deadline - 5:
            break
        
        pts = np.array(config)
        center0 = pts.mean(axis=0)
        
        # Try grid of angles and small center offsets
        for angle in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
            for dx in [-0.5, 0, 0.5]:
                for dy in [-0.5, 0, 0.5]:
                    c = (center0[0] + dx, center0[1] + dy)
                    s = enclosing_hex_side(pts, c, angle)
                    if s < best_side:
                        best_side = s
                        best_config = config
                        best_center = c
                        best_angle = angle
        
        # Local optimization from best grid point
        def objective(params):
            cx, cy, ang = params
            return enclosing_hex_side(pts, (cx, cy), ang)
        
        x0 = [best_center[0], best_center[1], best_angle]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_config = config
                best_center = (res.x[0], res.x[1])
                best_angle = res.x[2]
        except:
            pass
    
    if best_config is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
    
    inner = [[pt[0], pt[1], 0.0] for pt in best_config]
    return inner, (best_center[0], best_center[1]), best_side, best_angle
