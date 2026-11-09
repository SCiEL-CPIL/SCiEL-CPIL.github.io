import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def min_outer_side(vertices_all, outer_center, outer_angle_deg):
    outer_angle_rad = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = outer_angle_rad + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for v in vertices_all:
        rel = v - np.array(outer_center)
        projs = rel @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)

def overlap_penalty(verts1, verts2):
    normals = []
    for verts in [verts1, verts2]:
        for i in range(6):
            edge = verts[(i+1)%6] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals.append(normal / norm)
    
    for n in normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
            return 0.0
    
    min_gap = float('inf')
    for n in normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        gap = max(np.min(proj2) - np.max(proj1), np.min(proj1) - np.max(proj2))
        if gap < min_gap:
            min_gap = gap
    
    if min_gap < 0:
        return (-min_gap) ** 2 * 1000
    return 0.0

def objective(params, outer_center, outer_angle_deg):
    vertices_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        angle = params[3*i+2]
        verts = hexagon_vertices(cx, cy, 1.0, angle)
        vertices_list.append(verts)
    
    vertices_all = np.vstack(vertices_list)
    L = min_outer_side(vertices_all, outer_center, outer_angle_deg)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            penalty += overlap_penalty(vertices_list[i], vertices_list[j])
    
    return L + penalty

def construct():
    start_time = time.time()
    
    axial_positions = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (0,-2), (-2,0), (-2,1)
    ]
    
    initial_params = []
    for q, r in axial_positions:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        initial_params.extend([x, y, 0.0])
    initial_params = np.array(initial_params)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    def obj_func(p):
        return objective(p, outer_center, outer_angle)
    
    best_L = obj_func(initial_params)
    best_params = initial_params.copy()
    
    for restart in range(10):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            p0 = initial_params.copy()
        else:
            p0 = initial_params.copy()
            p0[:2*N] += np.random.normal(0, 0.5, 2*N)
            p0[2::3] = np.random.uniform(0, 60, N)
        
        try:
            result = minimize(obj_func, p0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_L:
                best_L = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    return inner, (0.0, 0.0), float(best_L), 0.0
