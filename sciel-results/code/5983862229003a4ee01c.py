import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    theta = np.deg2rad(angle_deg)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + theta
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hexagon_overlap(v1, v2, eps=1e-12):
    for i in range(6):
        e1 = v1[(i + 1) % 6] - v1[i]
        n1 = np.array([-e1[1], e1[0]])
        n1_norm = np.linalg.norm(n1)
        if n1_norm < 1e-15:
            continue
        n1 = n1 / n1_norm
        proj1 = np.dot(v1, n1)
        proj2 = np.dot(v2, n1)
        if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
            return False
    for i in range(6):
        e2 = v2[(i + 1) % 6] - v2[i]
        n2 = np.array([-e2[1], e2[0]])
        n2_norm = np.linalg.norm(n2)
        if n2_norm < 1e-15:
            continue
        n2 = n2 / n2_norm
        proj1 = np.dot(v1, n2)
        proj2 = np.dot(v2, n2)
        if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
            return False
    return True

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    theta = np.deg2rad(angle_deg)
    x = dx * np.cos(-theta) - dy * np.sin(-theta)
    y = dx * np.sin(-theta) + dy * np.cos(-theta)
    sqrt3 = np.sqrt(3)
    d = side * sqrt3 / 2
    for ang_deg in [30, 90, 150, 210, 270, 330]:
        ang = np.deg2rad(ang_deg)
        nx = np.cos(ang)
        ny = np.sin(ang)
        if x * nx + y * ny > d + 1e-12:
            return False
    return True

def _check_no_overlap(inner_pos, inner_angles):
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hex_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_angles[i])
            v2 = _hex_vertices(inner_pos[j][0], inner_pos[j][1], 1.0, inner_angles[j])
            if _hexagon_overlap(v1, v2):
                return False
    return True

def _compute_outer_side(inner_pos, inner_angles, outer_cx, outer_cy, outer_angle):
    sqrt3 = np.sqrt(3)
    theta = np.deg2rad(outer_angle)
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + theta
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    max_proj = 0.0
    for i in range(N):
        cx, cy, ang = inner_pos[i][0], inner_pos[i][1], inner_angles[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            dx = verts[j][0] - outer_cx
            dy = verts[j][1] - outer_cy
            for k in range(6):
                proj = dx * normals[k][0] + dy * normals[k][1]
                if proj > max_proj:
                    max_proj = proj
    s = max_proj * 2.0 / sqrt3
    return max(s, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    inner_pos = np.zeros((N, 2))
    inner_angles = np.zeros(N)
    
    # Initial hexagonal packing
    positions = [(0.0, 0.0)]
    for k in range(1, 7):
        angle = (k - 1) * 60.0
        r = 2.0
        positions.append((r * np.cos(np.deg2rad(angle)), r * np.sin(np.deg2rad(angle))))
    for k in range(7, 12):
        angle = (k - 7) * 60.0 + 30.0
        r = 4.0
        positions.append((r * np.cos(np.deg2rad(angle)), r * np.sin(np.deg2rad(angle))))
    
    for i in range(N):
        inner_pos[i] = positions[i]
        inner_angles[i] = 0.0
    
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    current_side = _compute_outer_side(inner_pos, inner_angles, outer_cx, outer_cy, outer_angle)
    best_side = current_side
    best_inner_pos = inner_pos.copy()
    best_inner_angles = inner_angles.copy()
    best_outer_cx, best_outer_cy = outer_cx, outer_cy
    best_outer_angle = outer_angle
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    step_size_pos = 0.5
    step_size_angle = 15.0
    step_size_outer = 0.3
    
    while time.time() < deadline - 5.0:
        r = np.random.rand()
        old_state = None
        
        if r < 0.5:
            i = np.random.randint(N)
            old_state = ('pos', i, inner_pos[i].copy())
            inner_pos[i][0] += np.random.normal(0, step_size_pos)
            inner_pos[i][1] += np.random.normal(0, step_size_pos)
        elif r < 0.7:
            i = np.random.randint(N)
            old_state = ('angle', i, inner_angles[i])
            inner_angles[i] += np.random.normal(0, step_size_angle)
        elif r < 0.85:
            old_state = ('outer_center', outer_cx, outer_cy)
            outer_cx += np.random.normal(0, step_size_outer)
            outer_cy += np.random.normal(0, step_size_outer)
        else:
            old_state = ('outer_angle', outer_angle)
            outer_angle += np.random.normal(0, step_size_angle)
        
        if not _check_no_overlap(inner_pos, inner_angles):
            # Revert
            if old_state[0] == 'pos':
                inner_pos[old_state[1]] = old_state[2]
            elif old_state[0] == 'angle':
                inner_angles[old_state[1]] = old_state[2]
            elif old_state[0] == 'outer_center':
                outer_cx, outer_cy = old_state[1], old_state[2]
            elif old_state[0] == 'outer_angle':
                outer_angle = old_state[1]
            T *= cooling
            if T < T_min:
                T = T_min
            continue
        
        new_side = _compute_outer_side(inner_pos, inner_angles, outer_cx, outer_cy, outer_angle)
        delta = new_side - current_side
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_inner_pos = inner_pos.copy()
                best_inner_angles = inner_angles.copy()
                best_outer_cx, best_outer_cy = outer_cx, outer_cy
                best_outer_angle = outer_angle
        else:
            # Revert
            if old_state[0] == 'pos':
                inner_pos[old_state[1]] = old_state[2]
            elif old_state[0] == 'angle':
                inner_angles[old_state[1]] = old_state[2]
            elif old_state[0] == 'outer_center':
                outer_cx, outer_cy = old_state[1], old_state[2]
            elif old_state[0] == 'outer_angle':
                outer_angle = old_state[1]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    inner = [[float(best_inner_pos[i][0]), float(best_inner_pos[i][1]), float(best_inner_angles[i])] for i in range(N)]
    return inner, (float(best_outer_cx), float(best_outer_cy)), float(best_side), float(best_outer_angle)
