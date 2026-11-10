import numpy as np
from scipy.optimize import minimize

def get_hex_vertices(c, s, angle_deg):
    cx, cy = c
    a = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = a + np.deg2rad(60 * i)
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def min_distance_hex_hex(H1, H2):
    def get_axes(H):
        axes = []
        for i in range(6):
            p1 = H[i]
            p2 = H[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            axes.append(n)
        return axes
    
    axes1 = get_axes(H1)
    axes2 = get_axes(H2)
    all_axes = axes1 + axes2
    
    overlap = True
    min_overlap = float('inf')
    
    for n in all_axes:
        proj1 = H1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = H2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            overlap = False
            break
        
        overlap_len = min(max1, max2) - max(min1, min2)
        if overlap_len < min_overlap:
            min_overlap = overlap_len
            
    if not overlap:
        min_dist = float('inf')
        for v in H1:
            for i in range(6):
                p1 = H2[i]
                p2 = H2[(i + 1) % 6]
                v_rel = v - p1
                e = p2 - p1
                e_len_sq = np.dot(e, e)
                if e_len_sq == 0: continue
                t = np.dot(v_rel, e) / e_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * e
                dist = np.linalg.norm(v - closest)
                if dist < min_dist: min_dist = dist
        for v in H2:
            for i in range(6):
                p1 = H1[i]
                p2 = H1[(i + 1) % 6]
                v_rel = v - p1
                e = p2 - p1
                e_len_sq = np.dot(e, e)
                if e_len_sq == 0: continue
                t = np.dot(v_rel, e) / e_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * e
                dist = np.linalg.norm(v - closest)
                if dist < min_dist: min_dist = dist
        return min_dist
    else:
        return -min_overlap

def hexagon_packing_12():
    n = 12
    best_score = float('inf')
    best_params = None
    
    # Run multiple restarts
    for _ in range(10):
        # Initial guess
        centers = []
        angles = []
        centers.append([0, 0])
        angles.append(0)
        d1 = np.sqrt(3)
        for i in range(6):
            theta = np.deg2rad(30 + 60 * i)
            centers.append([d1 * np.cos(theta), d1 * np.sin(theta)])
            angles.append(0)
        d2 = 2 * np.sqrt(3)
        for i in range(5):
            theta = np.deg2rad(0 + 72 * i)
            centers.append([d2 * np.cos(theta) + np.random.randn()*0.1, d2 * np.sin(theta) + np.random.randn()*0.1])
            angles.append(np.random.randn()*10)
            
        initial_params = []
        for c, a in zip(centers, angles):
            initial_params.extend([c[0], c[1], a])
            
        def objective(params):
            c = []
            a = []
            for i in range(n):
                c.append(params[3*i])
                c.append(params[3*i+1])
                a.append(params[3*i+2])
                
            # Outer angle fixed to 0
            outer_angle = 0
            outer_normals = []
            oa = np.deg2rad(outer_angle)
            for i in range(6):
                phi = oa + np.deg2rad(30 + 60 * i)
                outer_normals.append([np.cos(phi), np.sin(phi)])
            outer_normals = np.array(outer_normals)
            
            max_proj = 0
            for i in range(n):
                H = get_hex_vertices(np.array([c[2*i], c[2*i+1]]), 1, a[i])
                for v in H:
                    projs = v @ outer_normals.T
                    max_p = np.max(projs)
                    if max_p > max_proj:
                        max_proj = max_p
                        
            s_req = (2 / np.sqrt(3)) * max_proj
            
            penalty = 0
            hexagons = []
            for i in range(n):
                H = get_hex_vertices(np.array([c[2*i], c[2*i+1]]), 1, a[i])
                hexagons.append(H)
                
            for i in range(n):
                for j in range(i + 1, n):
                    dist = min_distance_hex_hex(hexagons[i], hexagons[j])
                    if dist < 0:
                        penalty += dist * dist
                        
            return s_req + 100 * penalty
            
        result = minimize(objective, initial_params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    # Extract best
    best_centers = []
    best_angles = []
    for i in range(n):
        best_centers.append([best_params[3*i], best_params[3*i+1]])
        best_angles.append(best_params[3*i+2])
        
    c = []
    a = []
    for i in range(n):
        c.append(best_params[3*i])
        c.append(best_params[3*i+1])
        a.append(best_params[3*i+2])
        
    outer_angle = 0
    outer_normals = []
    oa = np.deg2rad(outer_angle)
    for i in range(6):
        phi = oa + np.deg2rad(30 + 60 * i)
        outer_normals.append([np.cos(phi), np.sin(phi)])
    outer_normals = np.array(outer_normals)
    
    max_proj = 0
    for i in range(n):
        H = get_hex_vertices(np.array([c[2*i], c[2*i+1]]), 1, a[i])
        for v in H:
            projs = v @ outer_normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
                
    s_final = (2 / np.sqrt(3)) * max_proj
    
    inner_hex_data = np.array([[c[2*i], c[2*i+1], a[i]] for i in range(n)])
    outer_hex_data = np.array([0, 0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final
