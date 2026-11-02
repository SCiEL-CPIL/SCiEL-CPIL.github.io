import numpy as np
import time
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 145.0

def calc_S(points):
    verts = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2.0, 0.5],
        [math.sqrt(3)/2.0, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2.0, -0.5],
        [-math.sqrt(3)/2.0, 0.5]
    ])
    
    max_S = 0.0
    for i in range(len(points)):
        cx, cy = points[i]
        vx = cx + verts[:, 0]
        vy = cy + verts[:, 1]
        
        s1 = np.max(np.abs(vy))
        s2 = np.max(np.abs(vx + math.sqrt(3)*vy)) / math.sqrt(3)
        s3 = np.max(np.abs(-vx + math.sqrt(3)*vy)) / math.sqrt(3)
        
        current_S = max(s1, s2, s3)
        if current_S > max_S:
            max_S = current_S
    return max_S

def overlap_penalty(points):
    min_dist = math.sqrt(3)
    penalty = 0.0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                penalty += (min_dist - d)**2
    return penalty

def cost(params):
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e10
    points = params.reshape(-1, 2)
    S = calc_S(points)
    pen = overlap_penalty(points)
    return S + 100.0 * pen

def generate_guess(idx):
    # 3-3-3-2 grid
    pts = []
    for r in [-1, 0, 1, 2]:
        for q in range(-1, 2):
            if r == 2 and q == -1:
                continue
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append([x, y])
    pts = np.array(pts)
    
    # Rotate and shift for diversity
    angle = idx * math.pi / 6.0
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    # Small random shift
    pts += np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
    return pts.flatten()

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_S = 100.0
    best_params = None
    
    np.random.seed(42) # For reproducibility of restarts
    
    for i in range(10):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        initial_points = generate_guess(i)
        
        try:
            res = minimize(cost, initial_points, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_S and res.fun < 1e9:
                # Verify validity
                pts = res.x.reshape(-1, 2)
                if overlap_penalty(pts) < 1e-6:
                    best_S = calc_S(pts)
                    best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to seed-like or simple grid
        pts = generate_guess(0).reshape(-1, 2)
        best_params = pts.flatten()
        best_S = calc_S(pts)

    points = best_params.reshape(-1, 2)
    final_S = calc_S(points)
    
    inner = []
    for p in points:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), final_S, 0.0
# EVOLVE-BLOCK-END
