"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    np.random.seed(42)
    
    # Initial positions: Random in a small circle
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 2.0, N)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    rot = np.zeros(N)
    
    best_score = float('inf')
    best_x = x.copy()
    best_y = y.copy()
    best_rot = rot.copy()
    
    unit_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals for inner hexagon (flat top)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    base_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    def get_hex_vertices(cx, cy, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return unit_vertices @ R.T + np.array([cx, cy])
    
    def check_overlap(i, j, xs, ys, rots):
        dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
        if dist_sq > 4.0:
            return False
        
        vi = get_hex_vertices(xs[i], ys[i], rots[i])
        vj = get_hex_vertices(xs[j], ys[j], rots[j])
        
        ni = base_normals @ np.array([[np.cos(rots[i]), -np.sin(rots[i])], [np.sin(rots[i]), np.cos(rots[i])]]).T
        nj = base_normals @ np.array([[np.cos(rots[j]), -np.sin(rots[j])], [np.sin(rots[j]), np.cos(rots[j])]]).T
        
        for normals in [ni, nj]:
            for k in range(6):
                n = normals[k]
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                if np.max(proj_i) < np.min(proj_j) or np.max(proj_j) < np.min(proj_i):
                    return False
        return True

    def calculate_side_length(xs, ys, rots):
        all_vertices = []
        for i in range(N):
            all_vertices.append(get_hex_vertices(xs[i], ys[i], rots[i]))
        all_vertices = np.vstack(all_vertices)
        
        # Check two outer orientations: 0 and 30 degrees
        # Outer hexagon normals for 0 deg align (flat top) are at 30, 90...
        # Outer hexagon normals for 30 deg align (pointy top) are at 0, 60...
        
        min_side = float('inf')
        
        # Orientation 0: Normals at 30, 90, 150
        n0 = base_normals[0]
        n1 = base_normals[1]
        n2 = base_normals[2]
        w0 = np.max(np.dot(all_vertices, n0)) - np.min(np.dot(all_vertices, n0))
        w1 = np.max(np.dot(all_vertices, n1)) - np.min(np.dot(all_vertices, n1))
        w2 = np.max(np.dot(all_vertices, n2)) - np.min(np.dot(all_vertices, n2))
        side0 = max(w0, w1, w2) / np.sqrt(3)
        if side0 < min_side: min_side = side0
        
        # Orientation 30: Normals at 0, 60, 120
        # These are rotated by -30 degrees from base_normals
        rot_30 = np.deg2rad(-30)
        R_rot = np.array([[np.cos(rot_30), -np.sin(rot_30)], [np.sin(rot_30), np.cos(rot_30)]])
        n0_30 = base_normals[0] @ R_rot.T
        n1_30 = base_normals[1] @ R_rot.T
        n2_30 = base_normals[2] @ R_rot.T
        
        w0_30 = np.max(np.dot(all_vertices, n0_30)) - np.min(np.dot(all_vertices, n0_30))
        w1_30 = np.max(np.dot(all_vertices, n1_30)) - np.min(np.dot(all_vertices, n1_30))
        w2_30 = np.max(np.dot(all_vertices, n2_30)) - np.min(np.dot(all_vertices, n2_30))
        side30 = max(w0_30, w1_30, w2_30) / np.sqrt(3)
        if side30 < min_side: min_side = side30
        
        return min_side

    current_score = calculate_side_length(x, y, rot)
    if current_score < best_score:
        best_score = current_score
        best_x = x.copy()
        best_y = y.copy()
        best_rot = rot.copy()

    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if temp < 0.01:
            break
            
        i = np.random.randint(N)
        old_x, old_y, old_rot = x[i], y[i], rot[i]
        
        x[i] += np.random.normal(0, 0.5)
        y[i] += np.random.normal(0, 0.5)
        rot[i] += np.random.normal(0, 0.5)
        
        valid = True
        for j in range(N):
            if i == j: continue
            if check_overlap(i, j, x, y, rot):
                valid = False
                break
        
        if valid:
            new_score = calculate_side_length(x, y, rot)
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                    best_y = y.copy()
                    best_rot = rot.copy()
            else:
                x[i], y[i], rot[i] = old_x, old_y, old_rot
        else:
            x[i], y[i], rot[i] = old_x, old_y, old_rot
            
        temp *= cooling

    inner = []
    for i in range(N):
        inner.append([best_x[i], best_y[i], np.rad2deg(best_rot[i])])
    
    return inner, (0.0, 0.0), best_score, 0.0
# EVOLVE-BLOCK-END
