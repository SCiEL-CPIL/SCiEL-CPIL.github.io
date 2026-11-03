import numpy as np
import time

N = 11

# Precompute base hexagon vertices (side 1, centered at origin, angle 0)
# Angles 0, 60, ...
_BASE_ANGLES = np.deg2rad([0, 60, 120, 180, 240, 300])
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

# Precompute rotation matrices? No, angles change.
# But we can compute cos/sin once per hex.

def get_hex_vertices(cx, cy, angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _BASE_VERTS @ R.T + np.array([cx, cy])

def check_overlap_penalty(verts1, verts2):
    # verts1, verts2: (6, 2) arrays
    # Returns min distance to separate (0 if not overlapping)
    # Using SAT
    
    def get_axes(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return normals / norms
    
    axes1 = get_axes(verts1)
    axes2 = get_axes(verts2)
    all_axes = np.vstack((axes1, axes2))
    
    min_overlap = float('inf')
    
    for axis in all_axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def objective(params):
    xs = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    all_verts = []
    for i in range(N):
        v = get_hex_vertices(xs[i], ys[i], angles[i])
        all_verts.append(v)
    
    # Outer Hexagon Side Length
    # Normals at 30, 90, ...
    # Precomputed normals
    # n_k = (cos(30+60k), sin(30+60k))
    # We can precompute this
    # Let's assume it's defined globally
    
    max_proj = 0.0
    for v in all_verts:
        projs = np.dot(v, _NORMALS.T)
        max_proj = max(max_proj, np.max(projs))
        
    S = max_proj / (np.sqrt(3) / 2)
    
    total_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ov = check_overlap_penalty(all_verts[i], all_verts[j])
            total_overlap += ov
            
    penalty = total_overlap * 100.0
    cost = S + penalty
    
    is_valid = total_overlap < 1e-6
    return cost, is_valid

# Precompute normals for outer hexagon
_NORMALS = []
for deg in [30, 90, 150, 210, 270, 330]:
    rad = np.deg2rad(deg)
    _NORMALS.append([np.cos(rad), np.sin(rad)])
_NORMALS = np.array(_NORMALS)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Configuration
    # Hexagonal cluster
    # 1 center, 6 neighbors, 4 more
    # Let's use a grid for simplicity and validity
    positions = [
        (0, 0),
        (2, 0),
        (4, 0),
        (6, 0),
        (1, 1.732),
        (3, 1.732),
        (5, 1.732),
        (0, 3.464),
        (2, 3.464),
        (4, 3.464),
        (6, 3.464)
    ]
    
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = positions[i][0]
        params[3*i+1] = positions[i][1]
        params[3*i+2] = 0.0
        
    cx = np.mean(params[0::3])
    cy = np.mean(params[1::3])
    params[0::3] -= cx
    params[1::3] -= cy
    
    current_cost, current_valid = objective(params)
    best_params = params.copy()
    best_cost = current_cost
    best_valid = current_valid
    
    T = 1.0
    step = 0.5
    
    while time.time() < deadline:
        new_params = params.copy()
        num_move = np.random.randint(1, 4)
        indices = np.random.choice(N, num_move, replace=False)
        for idx in indices:
            new_params[3*idx] += np.random.normal(0, step)
            new_params[3*idx+1] += np.random.normal(0, step)
            new_params[3*idx+2] += np.random.normal(0, step * 0.1)
            
        new_cost, new_valid = objective(new_params)
        
        if new_cost < current_cost:
            params = new_params
            current_cost = new_cost
            current_valid = new_valid
            if new_valid and (not best_valid or new_cost < best_cost):
                best_params = params.copy()
                best_cost = new_cost
                best_valid = True
        else:
            delta = new_cost - current_cost
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    params = new_params
                    current_cost = new_cost
                    current_valid = new_valid
                    if new_valid and (not best_valid or new_cost < best_cost):
                        best_params = params.copy()
                        best_cost = new_cost
                        best_valid = True
        
        T *= 0.99
        step *= 0.999
        if step < 0.001:
            step = 0.001
            T = max(T, 0.1)
            
    xs = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], np.rad2deg(angles[i])])
        
    S = best_cost
    
    return inner, (0.0, 0.0), S, 0.0
