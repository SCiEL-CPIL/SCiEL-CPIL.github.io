import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _min_enclosing_hexagon_side(points, n_angles=12):
    points = np.array(points)
    if len(points) == 0:
        return 0.0
    best_s = float('inf')
    for alpha_deg in range(0, 180, 180 // n_angles):
        alpha_rad = math.radians(alpha_deg)
        normals = np.array([[math.cos(alpha_rad + k * math.pi / 3), math.sin(alpha_rad + k * math.pi / 3)] for k in range(6)])
        projections = points @ normals.T
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        centroid = points.mean(axis=0)
        diffs = points - centroid
        proj_diffs = diffs @ normals.T
        max_ext = proj_diffs.max()
        s = max_ext / (math.sqrt(3) / 2)
        if s < best_s:
            best_s = s
    return best_s

def _get_all_vertices(pos, ang):
    verts = []
    for i in range(N):
        v = _hexagon_vertices(pos[i, 0], pos[i, 1], ang[i])
        verts.extend(v)
    return np.array(verts)

def _objective(pos, ang):
    verts = _get_all_vertices(pos, ang)
    return _min_enclosing_hexagon_side(verts)

def _find_optimal_enclosing(pos, ang):
    verts = _get_all_vertices(pos, ang)
    best_s = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0
    for alpha_deg in range(0, 180, 3):
        alpha_rad = math.radians(alpha_deg)
        normals = np.array([[math.cos(alpha_rad + k * math.pi / 3), math.sin(alpha_rad + k * math.pi / 3)] for k in range(6)])
        projections = verts @ normals.T
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        midpoints = (max_proj + min_proj) / 2
        c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
        diffs = verts - c
        proj_diffs = diffs @ normals.T
        max_ext = proj_diffs.max()
        s = max_ext / (math.sqrt(3) / 2)
        if s < best_s:
            best_s = s
            best_center = (float(c[0]), float(c[1]))
            best_angle = alpha_deg
    return best_s, best_center, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Initial: 4x3 hexagonal grid
    dx = 1.75
    dy = 1.55
    positions = []
    angles = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y])
            angles.append(0.0)
    
    current_pos = np.array(positions)
    current_ang = np.array(angles)
    current_obj = _objective(current_pos, current_ang)
    
    best_pos = current_pos.copy()
    best_ang = current_ang.copy()
    best_obj = current_obj
    
    T = 0.4
    T_min = 0.0005
    cooling = 0.997
    
    while time.time() < deadline:
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, n_perturb, replace=False)
        for idx in indices:
            new_pos[idx, 0] += np.random.normal(0, T * 0.4)
            new_pos[idx, 1] += np.random.normal(0, T * 0.4)
            new_ang[idx] += np.random.normal(0, T * 25)
        
        new_obj = _objective(new_pos, new_ang)
        
        if new_obj < current_obj or (T > T_min and np.random.random() < math.exp((current_obj - new_obj) / T)):
            current_pos = new_pos
            current_ang = new_ang
            current_obj = new_obj
            if current_obj < best_obj:
                best_pos = current_pos.copy()
                best_ang = current_ang.copy()
                best_obj = current_obj
        
        T = max(T * cooling, T_min)
        if T <= T_min * 1.1:
            T = T_min * 30
    
    best_s, best_center, best_angle = _find_optimal_enclosing(best_pos, best_ang)
    
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])])
    
    return inner, best_center, float(best_s), float(best_angle)
# EVOLVE-BLOCK-END
