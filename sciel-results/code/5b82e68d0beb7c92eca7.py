import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600  # seconds
    
    n_hex = 11
    inner_side = 1.0
    eps = 1e-6
    
    # Containment normals for outer hexagon (centered at origin, angle 0)
    # Vertices at 0, 60, ...
    # Edge normals at 30, 90, 150
    containment_normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))]
    ])
    
    def get_vertices(x, y, theta):
        angles = theta + np.radians(60 * np.arange(6))
        return np.column_stack((x + inner_side * np.cos(angles), 
                                 y + inner_side * np.sin(angles)))
        
    def project_interval(vertices, normal):
        proj = vertices @ normal
        return np.min(proj), np.max(proj)
        
    def min_gap(vertices1, vertices2):
        # Compute normals for both hexagons
        def get_normals(vertices):
            normals = []
            for i in range(6):
                p1 = vertices[i]
                p2 = vertices[(i + 1) % 6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                n = n / np.linalg.norm(n)
                normals.append(n)
            return np.array(normals)
            
        n1 = get_normals(vertices1)
        n2 = get_normals(vertices2)
        
        all_normals = np.vstack([n1, n2])
        
        min_g = np.inf
        for n in all_normals:
            min1, max1 = project_interval(vertices1, n)
            min2, max2 = project_interval(vertices2, n)
            
            # Calculate gap
            # If disjoint, gap > 0
            # If overlapping, gap < 0
            gap_val = min(max1, max2) - max(min1, min2)
            
            if gap_val < min_g:
                min_g = gap_val
                
        return min_g

    def objective(params):
        max_proj = 0.0
        vertices_list = []
        
        for i in range(n_hex):
            x, y, theta = params[i]
            verts = get_vertices(x, y, theta)
            vertices_list.append(verts)
            
            for n in containment_normals:
                min_p, max_p = project_interval(verts, n)
                local_max = max(abs(min_p), abs(max_p))
                if local_max > max_proj:
                    max_proj = local_max
                    
        S = (2.0 / np.sqrt(3.0)) * max_proj
        
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                gap = min_gap(vertices_list[i], vertices_list[j])
                if gap < eps:
                    violation = eps - gap
                    penalty += violation * violation
        
        weight = 100.0
        return S + weight * penalty

    def initial_guess_1():
        params = np.zeros((n_hex, 3))
        params[0] = [0, 0, 0]
        r = np.sqrt(3)
        for k in range(1, 7):
            angle = np.radians(60 * (k - 1))
            params[k] = [r * np.cos(angle), r * np.sin(angle), 0]
        r2 = 2 * np.sqrt(3)
        angles = [30, 90, 150, 210]
        for idx, ang in enumerate(angles):
            params[7 + idx] = [r2 * np.cos(np.radians(ang)), r2 * np.sin(np.radians(ang)), 0]
        return params

    def initial_guess_2():
        params = np.random.rand(n_hex, 3) * 10 - 5
        params[:, 2] = np.random.rand(n_hex) * np.pi
        return params

    best_score = np.inf
    best_params = None
    
    # Run optimization
    for i in range(10):
        if time.time() - start_time > deadline - 60:
            break
            
        if i == 0:
            x0 = initial_guess_1()
        else:
            x0 = initial_guess_2()
            
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        
        if res.success:
            score = res.fun
            if score < best_score:
                best_score = score
                best_params = res.x
                
    if best_params is None:
        best_params = initial_guess_1()
        
    # Calculate final side length
    max_proj = 0.0
    for i in range(n_hex):
        x, y, theta = best_params[i]
        verts = get_vertices(x, y, theta)
        for n in containment_normals:
            min_p, max_p = project_interval(verts, n)
            local_max = max(abs(min_p), abs(max_p))
            if local_max > max_proj:
                max_proj = local_max
    side = (2.0 / np.sqrt(3.0)) * max_proj
        
    inner_hex_data = np.zeros((n_hex, 3))
    for i in range(n_hex):
        inner_hex_data[i, 0] = best_params[i, 0]
        inner_hex_data[i, 1] = best_params[i, 1]
        inner_hex_data[i, 2] = np.degrees(best_params[i, 2])
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, side
