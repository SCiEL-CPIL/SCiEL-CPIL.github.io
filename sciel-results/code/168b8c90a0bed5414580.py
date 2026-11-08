import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(center, angle_deg=0.0, side=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) + theta
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def point_in_hex(p, hex_verts):
    sign = None
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        cp = np.cross(p2 - p1, p - p1)
        if sign is None:
            sign = cp > 0
        elif (cp > 0) != sign:
            return False
    return True

def check_overlap(v1, v2, c1, c2):
    # c1, c2 are centers
    d = np.linalg.norm(c1 - c2)
    if d > 2.0:
        return False
    if d < np.sqrt(3) - 1e-9:
        return True
        
    # Detailed check
    for p in v1:
        if point_in_hex(p, v2):
            return True
    for p in v2:
        if point_in_hex(p, v1):
            return True
    return False

def calculate_side_length(positions, angles):
    normals = np.array([
        [1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    
    max_proj = 0.0
    all_verts = []
    for i in range(N):
        verts = get_hex_vertices(positions[i], angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    centroid = np.mean(all_verts, axis=0)
    all_verts_centered = all_verts - centroid
    
    for n in normals:
        projs = np.dot(all_verts_centered, n)
        max_p = np.max(projs)
        min_p = np.min(projs)
        req_s = (2.0 / np.sqrt(3.0)) * max(max_p, -min_p)
        if req_s > max_proj:
            max_proj = req_s
            
    return max_proj

def cost_function(params):
    positions = []
    angles = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        positions.append([x, y])
        angles.append(a)
        
    positions = np.array(positions)
    angles = np.array(angles)
    
    side = calculate_side_length(positions, angles)
    
    penalty = 0.0
    verts_list = [get_hex_vertices(positions[i], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(verts_list[i], verts_list[j], positions[i], positions[j]):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d)**2
                else:
                    penalty += 10.0
                    
    return side + 100.0 * penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_params = None
    
    fl = [
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5]
    ]
    extra = [
        [2*np.sqrt(3), 0.0],
        [np.sqrt(3), 3.0],
        [-np.sqrt(3), 3.0],
        [-2*np.sqrt(3), 0.0]
    ]
    
    init_positions = fl + extra
    init_angles = [0.0] * N
    
    init_params = []
    for i in range(N):
        init_params.append(init_positions[i][0])
        init_params.append(init_positions[i][1])
        init_params.append(init_angles[i])
        
    try:
        res = minimize(cost_function, init_params, method='Powell', options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if time.time() < deadline:
            params = res.x
            positions = []
            angles = []
            for i in range(N):
                positions.append([params[i*3], params[i*3+1]])
                angles.append(params[i*3+2])
            positions = np.array(positions)
            angles = np.array(angles)
            
            valid = True
            verts_list = [get_hex_vertices(positions[i], angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    if check_overlap(verts_list[i], verts_list[j], positions[i], positions[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                side = calculate_side_length(positions, angles)
                if side < best_side:
                    best_side = side
                    best_params = res.x
    except Exception:
        pass
        
    if best_params is None:
        positions = np.array(init_positions)
        angles = np.array(init_angles)
        valid = True
        verts_list = [get_hex_vertices(positions[i], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(verts_list[i], verts_list[j], positions[i], positions[j]):
                    valid = False
                    break
            if not valid: break
        
        if valid:
            best_side = calculate_side_length(positions, angles)
        else:
            best_side = 100.0
            positions = [[100.0 * k, 0.0] for k in range(N)]
            angles = [0.0] * N
            # Reconstruct params for return
            best_params = []
            for i in range(N):
                best_params.append(positions[i][0])
                best_params.append(positions[i][1])
                best_params.append(angles[i])
                
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
