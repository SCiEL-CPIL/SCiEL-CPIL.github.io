import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hexagon_vertices(x, y, angle, scale=1.0):
    angles = np.arange(6) * np.pi / 3.0 + angle
    rx = scale * np.cos(angles)
    ry = scale * np.sin(angles)
    return np.column_stack((rx + x, ry + y))

def min_distance_polygons(p1, p2):
    min_d = np.inf
    def point_to_segment_dist(p, a, b):
        pa = p - a
        ba = b - a
        denom = np.dot(ba, ba)
        if denom < 1e-12:
            return np.linalg.norm(pa)
        t = np.dot(pa, ba) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ba
        return np.linalg.norm(p - proj)
        
    for i in range(len(p1)):
        for j in range(len(p2)):
            d = point_to_segment_dist(p1[i], p2[j], p2[(j+1)%len(p2)])
            if d < min_d: min_d = d
            d = point_to_segment_dist(p2[i], p1[j], p1[(j+1)%len(p1)])
            if d < min_d: min_d = d
    return min_d

def container_violation(v, L, x_c, y_c, phi):
    vx = v[0] - x_c
    vy = v[1] - y_c
    cos_p = np.cos(-phi)
    sin_p = np.sin(-phi)
    lx = vx * cos_p - vy * sin_p
    ly = vx * sin_p + vy * cos_p
    
    r = L * np.sqrt(3) / 2.0
    # Normals for sides at 30, 90, 150, 210, 270, 330 degrees
    angles = np.array([np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    dists = normals.dot([lx, ly])
    viol = np.max(dists - r)
    if viol > 0:
        return viol * viol * 100.0
    return 0.0

def objective(params):
    x_c, y_c, phi = 0.0, 0.0, 0.0
    L = params[-1]
    if L < 0.1: L = 0.1
    
    penalty = 0.0
    hexes = []
    
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        th = params[i*3+2]
        verts = get_hexagon_vertices(x, y, th, 1.0)
        hexes.append(verts)
        
        for v in verts:
            p = container_violation(v, L, x_c, y_c, phi)
            penalty += p
            
    for i in range(N):
        for j in range(i+1, N):
            # Pruning: if center distance > 2, no overlap
            cx1, cy1 = params[i*3], params[i*3+1]
            cx2, cy2 = params[j*3], params[j*3+1]
            dc = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if dc > 2.0:
                continue
            p = check_overlap_penalty(hexes[i], hexes[j])
            penalty += p
            
    return L + penalty

def check_overlap_penalty(p1, p2):
    d = min_distance_polygons(p1, p2)
    if d < 0:
        return -d * -d * 100.0
    return 0.0

def construct():
    start_time = time.time()
    deadline = 145.0
    
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = np.sqrt(3) * (i + 0.5 * (j % 2))
            y = 1.5 * j
            candidates.append((x, y))
            
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = candidates[:N]
    
    init_params = []
    for x, y in init_positions:
        init_params.extend([x, y, 0.0])
    init_params.append(4.0)
    
    best_params = init_params
    best_val = objective(init_params)
    
    n_starts = 10
    np.random.seed(42)
    
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        if k == 0:
            current = np.array(init_params)
        else:
            current = np.array(init_params)
            current[:N*3] += np.random.normal(0, 0.5, N*3)
            current[-1] = np.random.uniform(3.5, 5.0)
            
        res = minimize(objective, current, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    L = best_params[-1]
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        th_deg = np.degrees(best_params[i*3+2])
        inner.append([x, y, th_deg])
        
    return inner, (0.0, 0.0), L, 0.0
