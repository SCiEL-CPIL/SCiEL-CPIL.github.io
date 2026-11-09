# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n_inner = 12
    
    def get_hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.pi/6 * np.arange(6)
        xs = cx + side * np.cos(angles)
        ys = cy + side * np.sin(angles)
        return np.column_stack((xs, ys))

    def point_segment_distance(p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def min_hex_distance(hex1_verts, hex2_verts):
        min_d = np.inf
        for v in hex1_verts:
            for i in range(6):
                a = hex2_verts[i]
                b = hex2_verts[(i+1)%6]
                d = point_segment_distance(v, a, b)
                if d < min_d:
                    min_d = d
        for v in hex2_verts:
            for i in range(6):
                a = hex1_verts[i]
                b = hex1_verts[(i+1)%6]
                d = point_segment_distance(v, a, b)
                if d < min_d:
                    min_d = d
        return min_d

    def objective(params):
        inner_params = params[:36].reshape(12, 3)
        S = params[36]
        outer_angle = params[37]
        
        if S < np.sqrt(12) - 1e-6:
            return 1e6 + (np.sqrt(12) - S) * 100
        
        inner_verts_list = []
        for i in range(12):
            cx, cy, theta = inner_params[i]
            verts = get_hex_vertices(cx, cy, 1.0, theta)
            inner_verts_list.append(verts)
        
        outer_verts = get_hex_vertices(0, 0, S, outer_angle)
        
        penalty = 0.0
        
        # Containment
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1)%6]
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0: continue
            for j in range(12):
                for v in inner_verts_list[j]:
                    rel_v = v - p1
                    cross_val = edge_vec[0]*rel_v[1] - edge_vec[1]*rel_v[0]
                    if cross_val < 0:
                        dist = -cross_val / edge_len
                        penalty += dist * dist
        
        # Non-overlap
        for i in range(12):
            for j in range(i+1, 12):
                d = min_hex_distance(inner_verts_list[i], inner_verts_list[j])
                if d < 0:
                    penalty += (-d) * (-d) * 10
        
        return S + penalty

    # Initial guess: Hexagonal lattice
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * np.sqrt(3) + j * np.sqrt(3)/2
            y = j * 1.5
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_points = lattice_points[:12]
    
    initial_params = np.zeros(38)
    for i in range(12):
        initial_params[i*3] = initial_points[i][0]
        initial_params[i*3+1] = initial_points[i][1]
        initial_params[i*3+2] = 0.0
    initial_params[36] = 5.0
    initial_params[37] = 0.0
    
    best_result = None
    best_score = -np.inf
    start_time = time.time()
    timeout = 590
    
    # Optimize
    try:
        res = minimize(objective, initial_params, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-7, 'gtol': 1e-7})
        if res.success:
            best_params = res.x
            score = 1.0 / best_params[36]
            if score > best_score:
                best_score = score
                best_result = best_params
    except Exception:
        pass
    
    if time.time() - start_time < timeout - 30:
        # Try Nelder-Mead on the best found so far
        if best_result is not None:
            try:
                res2 = minimize(objective, best_result, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res2.success:
                    score = 1.0 / res2.x[36]
                    if score > best_score:
                        best_score = score
                        best_result = res2.x
            except Exception:
                pass

    if best_result is None:
        best_params = initial_params
        max_dist = 0
        for i in range(12):
            cx, cy, theta = initial_params[i*3], initial_params[i*3+1], initial_params[i*3+2]
            verts = get_hex_vertices(cx, cy, 1.0, theta)
            for v in verts:
                d = np.linalg.norm(v)
                if d > max_dist:
                    max_dist = d
        best_params[36] = max_dist + 0.1
        best_params[37] = 0.0
    else:
        best_params = best_result

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[i*3]
        inner_hex_data[i, 1] = best_params[i*3+1]
        inner_hex_data[i, 2] = np.degrees(best_params[i*3+2])
        
    outer_hex_data = np.array([0, 0, np.degrees(best_params[37])])
    outer_hex_side_length = best_params[36]
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
