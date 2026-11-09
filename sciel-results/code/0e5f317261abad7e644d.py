import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3.0)
    
    # Initial positions from the current static packing
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:12]
    
    scale = 1.001
    angle_rad = math.pi / 6
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    p0 = np.zeros((N, 2))
    for i, (x, y) in enumerate(selected_centers):
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        p0[i] = [x_new * scale, y_new * scale]

    # Vertices of a unit hexagon with angle 0
    angles = np.array([k * np.pi / 3.0 for k in range(6)])
    verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Objective: Maximize the max distance squared
    def objective(p):
        diff = p[:, np.newaxis, :] + verts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        return np.max(dist_sq)

    # Constraints: Non-overlap
    eps = 1e-4
    min_dist_sq = (sqrt3 + eps)**2
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def con(p):
                    diff = p[i] - p[j]
                    return min_dist_sq - np.dot(diff, diff)
                return con
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    bounds = [(-5.0, 5.0)] * (N * 2)

    p_opt = p0
    side = (2.0 / sqrt3) * math.sqrt(objective(p0))
    
    try:
        res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success:
            p_opt = res.x.reshape(N, 2)
            max_dist_sq = res.fun
            max_dist = math.sqrt(max_dist_sq)
            side = (2.0 / sqrt3) * max_dist
    except Exception:
        pass

    inner = []
    for i in range(N):
        inner.append([p_opt[i, 0], p_opt[i, 1], 0.0])
        
    outer_center = (0.0, 0.0)
    angle = 0.0
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
