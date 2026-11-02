import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY_MARGIN = 8.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 10.0, 0.0

    # Precompute unit hexagon vertices for angle 0
    unit_hex_verts = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def get_rotated_vertices(angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return unit_hex_verts @ R.T

    def check_overlap(pos, angs_rad):
        # Fast rejection + vertex-in-hexagon test for all pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist > 2.0 * math.sqrt(3) / 2 * 1.5:  # generous bound
                    continue
                
                # Compute Minkowski difference: d must be outside hex_i + hex_j
                # For two unit hexagons with angles a_i, a_j, the Minkowski sum is a convex polygon
                # with up to 12 vertices. If d is inside this polygon, they overlap.
                # For same orientation: Minkowski sum is a hexagon with apothem 2*APOTHEM
                # For different orientations: more complex. Use vertex-in-poly check.
                
                v1 = pos[i] + get_rotated_vertices(angs_rad[i])
                v2 = pos[j] + get_rotated_vertices(angs_rad[j])
                
                # Check if any vertex of v1 is inside v2 or vice versa
                n2 = angs_rad[j] + math.pi/6 + np.arange(6) * (math.pi/3)
                n2_vec = np.column_stack((np.cos(n2), np.sin(n2)))
                diff1 = v1 - pos[j]
                proj1 = diff1 @ n2_vec.T
                if np.any(np.max(proj1, axis=1) <= APOTHEM + 1e-9):
                    return True
                
                n1 = angs_rad[i] + math.pi/6 + np.arange(6) * (math.pi/3)
                n1_vec = np.column_stack((np.cos(n1), np.sin(n1)))
                diff2 = v2 - pos[i]
                proj2 = diff2 @ n1_vec.T
                if np.any(np.max(proj2, axis=1) <= APOTHEM + 1e-9):
                    return True
        return False

    def calc_outer_size(pos, outer_rot_rad):
        # Minimal outer hexagon side length containing all vertices, with given rotation
        all_verts = np.vstack([pos[i] + get_rotated_vertices(0.0) for i in range(N)])
        n_outer = outer_rot_rad + math.pi/6 + np.arange(6) * (math.pi/3)
        n_outer_vec = np.column_stack((np.cos(n_outer), np.sin(n_outer)))
        
        widths = []
        for k in range(6):
            proj = all_verts @ n_outer_vec[k]
            w = np.max(proj) - np.min(proj)
            widths.append(w)
        max_w = max(widths)
        S = max_w / (2.0 * APOTHEM)
        C = np.mean(all_verts, axis=0)
        return S, C

    # Initial configuration: honeycomb-like
    pos = np.zeros((N, 2))
    pos[0] = [0, 0]
    for i in range(6):
        a = i * math.pi / 3
        pos[i+1] = [math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)]
    for i in range(4):
        a = math.pi/6 + i * math.pi / 2
        pos[i+7] = [2.0 * math.sqrt(3) * math.cos(a), 2.0 * math.sqrt(3) * math.sin(a)]
    
    best_S, best_C = calc_outer_size(pos, 0.0)
    if check_overlap(pos, np.zeros(N)):
        best_S = 100.0
    
    best_pos = pos.copy()
    best_rot = 0.0
    
    T = 1.0
    step_size = 0.3
    
    for t in range(8000):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pos_new = best_pos.copy()
        idx = np.random.randint(N)
        pos_new[idx] += np.random.normal(0, step_size, 2)
        rot_new = 0.0
        
        if np.random.rand() < 0.05:
            rot_new = best_rot + np.random.normal(0, 0.05)
        
        S_new, C_new = calc_outer_size(pos_new, rot_new)
        
        if check_overlap(pos_new, np.zeros(N)):
            S_new = 1000.0
            
        if S_new < best_S:
            best_S = S_new
            best_pos = pos_new.copy()
            best_rot = rot_new
        elif np.random.rand() < math.exp(-(S_new - best_S) / T):
            best_pos = pos_new.copy()
            best_rot = rot_new
            
        T = max(0.01, T * 0.995)
        if t % 200 == 0:
            step_size = max(0.05, step_size * 0.95)
    
    center = [float(best_C[0]), float(best_C[1])]
    side = float(best_S)
    rotation = float(best_rot)
    
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), 0.0])
    
    return inner, center, side, rotation
