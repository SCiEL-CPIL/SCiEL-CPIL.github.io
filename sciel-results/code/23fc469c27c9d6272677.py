import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, angle, size=1.0):
    angles = np.arange(6) * math.pi / 3 + angle
    verts = np.column_stack([size * np.cos(angles), size * np.sin(angles)])
    return verts + np.array([cx, cy])

def _get_all_points(centers, angles):
    base_angles = np.arange(6) * math.pi / 3
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    c = np.cos(angles)
    s = np.sin(angles)
    
    x_base = base_verts[:, 0]
    y_base = base_verts[:, 1]
    
    x_rot = c[:, None] * x_base[None, :] - s[:, None] * y_base[None, :]
    y_rot = s[:, None] * x_base[None, :] + c[:, None] * y_base[None, :]
    
    x_rot += centers[:, 0:1]
    y_rot += centers[:, 1:2]
    
    points = np.column_stack([x_rot.flatten(), y_rot.flatten()])
    return points

def _min_outer_side(points, outer_center, outer_angle):
    p = points - outer_center
    c = math.cos(-outer_angle)
    s = math.sin(-outer_angle)
    x = p[:, 0] * c - p[:, 1] * s
    y = p[:, 0] * s + p[:, 1] * c
    
    normals = np.array([
        [1, 0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1, 0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    dots = np.column_stack([x, y]) @ normals.T
    abs_dots = np.abs(dots)
    max_proj = np.max(abs_dots)
    
    return (2.0 / math.sqrt(3)) * max_proj

def _poly_overlap(v1, v2, eps=1e-9):
    polys = [v1, v2]
    for v in polys:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
    return True

def _overlap_penalty(centers, angles):
    penalty = 0.0
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(centers[i, 0], centers[i, 1], angles[i]))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 2.0:
                continue
            
            if _poly_overlap(verts_list[i], verts_list[j]):
                penalty += 10.0 * (2.0 - dist) ** 2 + 5.0
    return penalty

def _objective(params):
    centers = params[0:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    outer_center = params[3*N:3*N+2]
    outer_angle = params[3*N+2]
    
    points = _get_all_points(centers, angles)
    S = _min_outer_side(points, outer_center, outer_angle)
    
    pen = _overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _make_seed_grid():
    d = math.sqrt(3)
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    positions = [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    
    for i, (q, r) in enumerate(positions):
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        centers[i] = [x, y]
        
    return centers, angles

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0
    
    best_S = float('inf')
    best_params = None
    best_valid_params = None
    best_valid_S = float('inf')
    
    centers, angles = _make_seed_grid()
    angles = np.random.normal(0, 0.1, N)
    
    outer_center = np.mean(centers, axis=0)
    outer_angle = 0.0
    
    params = np.concatenate([centers.flatten(), angles, outer_center, [outer_angle]])
    
    T = 0.5
    cooling = 0.9995
    step_size = 0.2
    max_iter = 20000
    
    current_params = params.copy()
    current_cost = _objective(current_params)
    
    # Check if initial is valid
    pen = _overlap_penalty(current_params[0:2*N].reshape(N, 2), current_params[2*N:3*N])
    if pen < 1e-6:
        S = _min_outer_side(_get_all_points(current_params[0:2*N].reshape(N, 2), current_params[2*N:3*N]), current_params[3*N:3*N+2], current_params[3*N+2])
        best_valid_S = S
        best_valid_params = current_params.copy()
        best_S = S
        best_params = current_params.copy()

    for i in range(max_iter):
        if time.time() > deadline - safety_margin:
            break
            
        new_params = current_params + np.random.normal(0, step_size, len(current_params))
        new_params[2*N:3*N] = np.mod(new_params[2*N:3*N] + math.pi, 2*math.pi) - math.pi
        
        new_cost = _objective(new_params)
        
        if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
            current_params = new_params
            current_cost = new_cost
            
            pen = _overlap_penalty(current_params[0:2*N].reshape(N, 2), current_params[2*N:3*N])
            if pen < 1e-6:
                S = _min_outer_side(_get_all_points(current_params[0:2*N].reshape(N, 2), current_params[2*N:3*N]), current_params[3*N:3*N+2], current_params[3*N+2])
                if S < best_valid_S:
                    best_valid_S = S
                    best_valid_params = current_params.copy()
                    best_S = S
                    best_params = current_params.copy()
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        if i % 1000 == 0:
            step_size *= 0.9

    # Polish best valid params
    if best_valid_params is not None and time.time() < deadline - safety_margin:
        try:
            # Define a constrained objective for polishing? 
            # Just minimize S, and if penalty becomes > 0, reject.
            # L-BFGS-B doesn't handle constraints well without bounds.
            # We can just run L-BFGS-B on the full objective and check validity at the end.
            res = minimize(_objective, best_valid_params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_valid_S:
                pen = _overlap_penalty(res.x[0:2*N].reshape(N, 2), res.x[2*N:3*N])
                if pen < 1e-6:
                    best_valid_S = res.fun
                    best_valid_params = res.x
                    best_S = res.fun
                    best_params = res.x
        except Exception:
            pass
            
    if best_valid_params is not None:
        centers = best_valid_params[0:2*N].reshape(N, 2)
        angles = best_valid_params[2*N:3*N]
        outer_center = best_valid_params[3*N:3*N+2]
        outer_angle = best_valid_params[3*N+2]
        
        inner = []
        for i in range(N):
            inner.append([centers[i, 0], centers[i, 1], math.degrees(angles[i])])
            
        return inner, (outer_center[0], outer_center[1]), best_valid_S, math.degrees(outer_angle)
    else:
        # Fallback to seed
        centers = best_params[0:2*N].reshape(N, 2)
        angles = best_params[2*N:3*N]
        outer_center = best_params[3*N:3*N+2]
        outer_angle = best_params[3*N+2]
        inner = []
        for i in range(N):
            inner.append([centers[i, 0], centers[i, 1], math.degrees(angles[i])])
        return inner, (outer_center[0], outer_center[1]), best_S, math.degrees(outer_angle)

