import numpy as np
from scipy.optimize import minimize

N = 11

def get_inner_hex_vertices(cx, cy, angle):
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def get_outer_hex_vertices(S, angle):
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = S * np.cos(angles)
    y = S * np.sin(angles)
    return np.column_stack((x, y))

def point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def smooth_min(x, T=0.1):
    m = np.min(x)
    return m - (1.0 / T) * np.log(np.sum(np.exp(-(x - m) / T)))

def min_distance_between_hexagons(verts1, verts2, T=0.1):
    d1 = []
    d2 = []
    edges1 = np.column_stack((verts1, np.roll(verts1, -1, axis=0)))
    edges2 = np.column_stack((verts2, np.roll(verts2, -1, axis=0)))
    
    for i in range(6):
        p = verts1[i]
        for j in range(6):
            d1.append(point_to_segment_distance(p, edges2[j, 0], edges2[j, 1]))
            
    for i in range(6):
        p = verts2[i]
        for j in range(6):
            d2.append(point_to_segment_distance(p, edges1[j, 0], edges1[j, 1]))
            
    all_dists = np.array(d1 + d2)
    return smooth_min(all_dists, T)

def min_distance_to_outer_boundary(verts, outer_verts, T=0.1):
    d = []
    edges_outer = np.column_stack((outer_verts, np.roll(outer_verts, -1, axis=0)))
    
    for i in range(6):
        p = verts[i]
        for j in range(6):
            d.append(point_to_segment_distance(p, edges_outer[j, 0], edges_outer[j, 1]))
            
    all_dists = np.array(d)
    return smooth_min(all_dists, T)

def construct():
    def objective(params):
        S = params[-2]
        outer_angle = params[-1]
        
        if S < 1.0:
            S = 1.0
            
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        
        inner_verts = []
        for i in range(N):
            v = get_inner_hex_vertices(centers_x[i], centers_y[i], angles[i])
            inner_verts.append(v)
            
        outer_verts = get_outer_hex_vertices(S, outer_angle)
        
        penalty = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_between_hexagons(inner_verts[i], inner_verts[j], T=0.05)
                if d < 0:
                    penalty += d * d
                    
        for i in range(N):
            d = min_distance_to_outer_boundary(inner_verts[i], outer_verts, T=0.05)
            if d < 0:
                penalty += d * d
        
        return S + 1000.0 * penalty

    best_S = float('inf')
    best_params = None
    
    num_starts = 10
    
    for start_idx in range(num_starts):
        if start_idx == 0:
            d = np.sqrt(3) + 0.01
            init_centers_x = [0.0]
            init_centers_y = [0.0]
            init_angles = [0.0]
            
            for k in range(6):
                ang = k * np.pi / 3
                init_centers_x.append(d * np.cos(ang))
                init_centers_y.append(d * np.sin(ang))
                init_angles.append(0.0)
                
            for k in range(4):
                ang = (k * 60 + 30) * np.pi / 180
                init_centers_x.append(3.0 * np.cos(ang))
                init_centers_y.append(3.0 * np.sin(ang))
                init_angles.append(0.0)
                
            x0 = np.zeros(3 * N + 2)
            for i in range(N):
                x0[3*i] = init_centers_x[i]
                x0[3*i+1] = init_centers_y[i]
                x0[3*i+2] = init_angles[i]
            x0[-2] = 4.0
            x0[-1] = 0.0
        else:
            np.random.seed(start_idx + 100)
            x0 = np.zeros(3 * N + 2)
            for i in range(N):
                r = np.sqrt(np.random.random()) * 3.5
                theta = np.random.random() * 2 * np.pi
                x0[3*i] = r * np.cos(theta)
                x0[3*i+1] = r * np.sin(theta)
                x0[3*i+2] = np.random.random() * np.pi
            x0[-2] = 4.5
            x0[-1] = np.random.random() * np.pi

        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_S:
                S_val = res.x[-2]
                penalty_val = (res.fun - S_val) / 1000.0
                if penalty_val < 1e-4:
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, np.degrees(ang)])
    
    outer_center = (0.0, 0.0)
    outer_side = best_params[-2]
    outer_angle = best_params[-1]
    
    return inner, outer_center, outer_side, outer_angle
