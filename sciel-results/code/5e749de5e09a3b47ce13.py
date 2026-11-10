# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_valid(inner, outer, s, tol=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for i in range(6):
            edge_len = np.linalg.norm(verts[(i+1)%6] - verts[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            overlap = True
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    if max_i < min_j - tol or max_j < min_i - tol:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False
    return True

def _compute_s(inner, oang):
    """Compute minimum side length of outer hexagon containing all inner hexagons."""
    a = np.radians(oang)
    
    all_verts = []
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    normals = []
    for k in range(6):
        theta = a + np.radians(60*k + 30)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    dots = all_verts @ normals.T
    max_dots = dots.max(axis=0)
    
    s_min = np.max(max_dots) / (np.sqrt(3)/2)
    return s_min

def _axial_to_cartesian(q, r):
    """Convert axial hex coordinates to cartesian for flat-top hexagons."""
    x = 1.5 * q
    y = np.sqrt(3) * (r + q/2.0)
    return x, y

def _get_initial_configs():
    """Generate multiple initial configurations for 12 hexagons."""
    configs = []
    
    # Config 1: Compact spiral
    hex_pos1 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(2,-1),(2,-2),(1,-2),(0,-2)]
    centers1 = np.array([_axial_to_cartesian(q,r) for q,r in hex_pos1])
    centers1 = centers1 - centers1.mean(axis=0)
    configs.append(centers1)
    
    # Config 2: 3-4-3-2
    hex_pos2 = [(0,0),(1,0),(-1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1),(2,0),(-2,0),(0,2)]
    centers2 = np.array([_axial_to_cartesian(q,r) for q,r in hex_pos2])
    centers2 = centers2 - centers2.mean(axis=0)
    configs.append(centers2)
    
    # Config 3: 4-4-4 rectangular
    hex_pos3 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,-1),(1,-1),(2,-1),(3,-1)]
    centers3 = np.array([_axial_to_cartesian(q,r) for q,r in hex_pos3])
    centers3 = centers3 - centers3.mean(axis=0)
    configs.append(centers3)
    
    # Config 4: 2-4-2-4
    hex_pos4 = [(0,0),(1,0),(0,1),(1,1),(0,-1),(1,-1),(2,0),(3,0),(2,1),(3,1),(2,-1),(3,-1)]
    centers4 = np.array([_axial_to_cartesian(q,r) for q,r in hex_pos4])
    centers4 = centers4 - centers4.mean(axis=0)
    configs.append(centers4)
    
    return configs

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = float('inf')
    best_params = None
    
    configs = _get_initial_configs()
    
    for config_idx, init_centers in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        def objective(params, init_centers=init_centers):
            centers = params[:24].reshape(12, 2)
            oang = params[24]
            centroid = centers.mean(axis=0)
            centers = centers - centroid
            inner = np.column_stack([centers, np.zeros(12)])
            s = _compute_s(inner, oang)
            return s
        
        init_params = np.append(init_centers.flatten(), 0.0)
        
        result = minimize(objective, init_params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 10000})
        
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x.copy()
        
        if time.time() < deadline - 60:
            result2 = minimize(objective, best_params, method='Nelder-Mead', 
                             options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'maxfev': 5000})
            if result2.fun < best_s:
                best_s = result2.fun
                best_params = result2.x.copy()
    
    if best_params is None:
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8
        return inner, outer, s
    
    centers = best_params[:24].reshape(12, 2)
    oang = best_params[24]
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    
    inner = np.column_stack([centers, np.zeros(12)])
    outer = np.array([0.0, 0.0, oang])
    s = _compute_s(inner, oang)
    
    if not _check_valid(inner, outer, s):
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8
    
    return inner, outer, s

# EVOLVE-BLOCK-END
