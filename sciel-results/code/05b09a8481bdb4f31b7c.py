import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.deg2rad(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            theta = a + np.deg2rad(60 * i)
            verts[i, 0] = cx + side * np.cos(theta)
            verts[i, 1] = cy + side * np.sin(theta)
        return verts

    def calc_min_s(centers, angles, outer_angle_deg=0.0):
        all_verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        normals = []
        for k in range(6):
            theta = np.deg2rad(30 + 60 * k)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        
        s = (2.0 / np.sqrt(3)) * max_proj
        return s

    def get_overlap_penalty(centers, angles):
        penalty = 0.0
        # Precompute vertices for efficiency? No, angles change.
        verts_list = []
        for i in range(12):
            verts_list.append(get_hex_vertices(centers[i, 0], centers[i, 1], angles[i]))
            
        for i in range(12):
            v_i = verts_list[i]
            for j in range(i + 1, 12):
                v_j = verts_list[j]
                min_pen = float('inf')
                separated = False
                # Check all 12 axes (6 from i, 6 from j)
                for poly in [v_i, v_j]:
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm_len = np.linalg.norm(normal)
                        if norm_len < 1e-6: continue
                        normal = normal / norm_len
                        
                        proj_i = v_i @ normal
                        proj_j = v_j @ normal
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        # Separation condition: max_i < min_j or max_j < min_i
                        # Gap = min(max_i, max_j) - max(min_i, min_j)
                        # If Gap < 0, they overlap.
                        gap = min(max_i, max_j) - max(min_i, min_j)
                        
                        if gap < 0:
                            # Overlapping on this axis
                            if -gap < min_pen:
                                min_pen = -gap
                        else:
                            # Separated on this axis
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    # Overlap depth is min_pen (positive value representing penetration)
                    # Use a smooth penalty: square of penetration
                    penalty += min_pen ** 2
        return penalty

    def objective(params):
        centers = params[0:24].reshape(12, 2)
        angles = params[24:36]
        
        s = calc_min_s(centers, angles)
        pen = get_overlap_penalty(centers, angles)
        
        # High penalty weight to force non-overlap
        # The penalty is squared, so for small overlaps it's small.
        # We want to minimize s, but s is ~3-5. Penalties can be large if overlaps are bad.
        # Let's use a weight that makes overlap very costly.
        return s + 1000.0 * pen

    def is_valid(centers, angles, s):
        # Check overlaps strictly
        verts_list = []
        for i in range(12):
            verts_list.append(get_hex_vertices(centers[i, 0], centers[i, 1], angles[i]))
        
        for i in range(12):
            v_i = verts_list[i]
            for j in range(i + 1, 12):
                v_j = verts_list[j]
                separated = False
                for poly in [v_i, v_j]:
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm_len = np.linalg.norm(normal)
                        if norm_len < 1e-6: continue
                        normal = normal / norm_len
                        
                        proj_i = v_i @ normal
                        proj_j = v_j @ normal
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        
                        if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return False
        
        # Check containment
        # Outer hexagon vertices
        outer_verts = get_hex_vertices(0, 0, 0.0, s)
        
        for i in range(12):
            v_i = verts_list[i]
            for v in v_i:
                # Check if v is inside outer hexagon
                inside = True
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    # cross product to check if v is on the left of edge p1->p2 (assuming CCW)
                    # Outer hexagon vertices are generated CCW?
                    # cos/sin with increasing angle is CCW.
                    # So interior is to the left of each edge.
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0] * vec[1] - edge[1] * vec[0]
                    if cross < -1e-6:
                        inside = False
                        break
                if not inside:
                    return False
        return True

    # Initial Guess: Hexagonal lattice
    # 12 hexagons.
    # Arrangement:
    #   . X X X .
    #   X X X X X
    #   . X X X .
    # This is 3+5+4 = 12? No.
    # Let's use a compact hexagon of side 2 (7 hexes) + 5 more.
    # Or simply a grid.
    
    # A better initial guess:
    # Centers of a hexagonal grid.
    # dx = sqrt(3), dy = 1.5
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (3*sqrt(3)/2, 1.5), (-3*sqrt(3)/2, 1.5)
    # ...
    
    # Let's just place them in a 3x4 grid and let the optimizer fix it.
    # But a hexagonal cluster is better.
    
    centers_init = []
    angles_init = []
    
    # 1 center
    centers_init.append([0, 0])
    angles_init.append(0.0)
    
    # 6 around it
    for i in range(6):
        ang = np.deg2rad(60 * i)
        r = np.sqrt(3) # Distance between centers of touching hexes is sqrt(3) * side?
        # Side 1. Distance between centers = sqrt(3).
        centers_init.append([r * np.cos(ang), r * np.sin(ang)])
        angles_init.append(0.0)
        
    # 5 more
    # Place them around the ring
    # The ring has 6 hexes. We need 5 more.
    # Let's just add them in a line or spread out.
    for i in range(5):
        ang = np.deg2rad(60 * i + 30) # Offset
        r = 2 * np.sqrt(3) # Further out
        centers_init.append([r * np.cos(ang), r * np.sin(ang)])
        angles_init.append(0.0)
        
    centers_init = np.array(centers_init)
    angles_init = np.array(angles_init)
    
    params_init = np.concatenate([centers_init.flatten(), angles_init])
    
    best_params = params_init
    best_val = objective(params_init)
    best_valid_params = None
    best_valid_s = float('inf')
    
    # Check if initial is valid
    s_init = calc_min_s(centers_init, angles_init)
    if is_valid(centers_init, angles_init, s_init):
        best_valid_params = params_init.copy()
        best_valid_s = s_init

    # Optimize
    try:
        # Use L-BFGS-B
        res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
            
            s_res = calc_min_s(res.x[0:24].reshape(12, 2), res.x[24:36])
            if is_valid(res.x[0:24].reshape(12, 2), res.x[24:36], s_res):
                if s_res < best_valid_s:
                    best_valid_s = s_res
                    best_valid_params = res.x.copy()
    except Exception:
        pass

    # Random restarts
    np.random.seed(42)
    while time.time() - start_time < deadline - 30:
        # Perturb
        noise = np.random.normal(0, 0.5, size=params_init.shape)
        params_try = params_init + noise
        
        try:
            res = minimize(objective, params_try, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_val:
                best_params = res.x
                best_val = res.fun
                
                s_res = calc_min_s(res.x[0:24].reshape(12, 2), res.x[24:36])
                if is_valid(res.x[0:24].reshape(12, 2), res.x[24:36], s_res):
                    if s_res < best_valid_s:
                        best_valid_s = s_res
                        best_valid_params = res.x.copy()
        except Exception:
            continue

    # If we found a valid solution, use it
    if best_valid_params is not None:
        centers = best_valid_params[0:24].reshape(12, 2)
        angles = best_valid_params[24:36]
        s_final = calc_min_s(centers, angles)
        
        inner_hex_data = np.column_stack((centers, angles))
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, s_final

    # Fallback: Return the best optimized solution even if slightly invalid? 
    # No, task says invalid scores 0. 
    # We must return a valid one. 
    # If no valid one found, we have a problem. 
    # Let's try to repair the best_params if it's close.
    
    # For now, if no valid solution, return a very large valid one (e.g. spread out)
    # This ensures score > 0.
    centers_fallback = []
    angles_fallback = []
    for i in range(12):
        # Spread them far apart
        ang = 2 * np.pi * i / 12
        r = 10.0
        centers_fallback.append([r * np.cos(ang), r * np.sin(ang)])
        angles_fallback.append(0.0)
        
    centers_fallback = np.array(centers_fallback)
    angles_fallback = np.array(angles_fallback)
    
    s_fallback = calc_min_s(centers_fallback, angles_fallback)
    
    inner_hex_data = np.column_stack((centers_fallback, angles_fallback))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_fallback
