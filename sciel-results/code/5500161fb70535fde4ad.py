import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    min_dist = np.sqrt(3)
    
    def calc_S(pos):
        abs_x = np.abs(pos[:, 0])
        abs_y = np.abs(pos[:, 1])
        
        s1 = abs_x
        s2 = (2.0 / np.sqrt(3.0)) * abs_y
        s3 = abs_x + (1.0 / np.sqrt(3.0)) * abs_y
        
        s_per_point = np.maximum(s1, np.maximum(s2, s3))
        return np.max(s_per_point)
    
    def objective(x):
        pos = x.reshape(N, 2)
        return calc_S(pos)
    
    def make_constraints():
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def con(x, i=i, j=j):
                    pos = x.reshape(N, 2)
                    pi = pos[i]
                    pj = pos[j]
                    d_sq = np.sum((pi - pj)**2)
                    return d_sq - min_dist**2
                cons.append({'type': 'ineq', 'fun': con})
        return cons
    
    constraints = make_constraints()
    
    best_S = float('inf')
    best_pos = None
    
    # Honeycomb guess
    hex_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = min_dist * (q + r/2.0)
            y = min_dist * (np.sqrt(3)/2.0 * r)
            hex_points.append([x, y])
    
    hex_points = np.array(hex_points)
    dists = np.linalg.norm(hex_points, axis=1)
    indices = np.argsort(dists)
    init_honeycomb = hex_points[indices[:N]].copy()
    
    inits = [init_honeycomb]
    
    np.random.seed(42)
    for _ in range(30):
        pts = np.random.uniform(-4, 4, (N, 2))
        inits.append(pts)
        
    start_time = time.time()
    timeout = 110.0
    
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=None, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                pos = res.x.reshape(N, 2)
                S = res.fun
                if S < best_S:
                    best_S = S
                    best_pos = pos
        except Exception:
            pass
            
    if best_pos is None:
        best_pos = init_honeycomb
        best_S = calc_S(best_pos)
        
    inner = []
    for p in best_pos:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
