import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

# Precompute triangle indices
tri_idx = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            tri_idx.append((i, j, k))
tri_idx = np.array(tri_idx)

# Map each point to the indices of triangles it belongs to
point_tri_map = [[] for _ in range(N)]
for t_idx, (i, j, k) in enumerate(tri_idx):
    point_tri_map[i].append(t_idx)
    point_tri_map[j].append(t_idx)
    point_tri_map[k].append(t_idx)

def bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Triangle vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2).
    P = u*A + v*B + (1-u-v)*C
    """
    x = 0.5 - 0.5 * u + 0.5 * v
    y = H * (1 - u - v)
    return x, y

def params_to_points(params):
    """Convert 2N parameters (u, v for each point) to N Cartesian points."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        
        # Ensure valid barycentric coordinates
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        s = u + v
        if s > 1:
            u = u / s
            v = v / s
            
        x, y = bary_to_cart(u, v)
        points[i] = [x, y]
    return points

def compute_all_areas(points):
    """Compute areas of all triangles formed by N points."""
    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def objective(params):
    """Objective function for scipy.optimize: minimize negative min area."""
    points = params_to_points(params)
    areas = compute_all_areas(points)
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_params = np.zeros(2 * N)
    best_score = -np.inf
    
    # Initial random configurations
    for _ in range(20):
        params = np.random.uniform(0, 1, size=2 * N)
        for i in range(N):
            u = params[2*i]
            v = params[2*i+1]
            s = u + v
            if s > 1:
                params[2*i] = u / s
                params[2*i+1] = v / s
        score = -objective(params)
        if score > best_score:
            best_score = score
            best_params = params.copy()
            
    current_params = best_params.copy()
    points = params_to_points(current_params)
    areas = compute_all_areas(points)
    current_min_area = np.min(areas)
    
    T_start = 0.1
    T_end = 0.0001
    num_steps = 1000000
    
    for step in range(num_steps):
        if time.time() > deadline - 5.0:
            break
            
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        k = rng.integers(N)
        
        old_u = current_params[2 * k]
        old_v = current_params[2 * k + 1]
        old_x, old_y = bary_to_cart(old_u, old_v)
        
        delta_u = rng.normal(0, 0.05)
        delta_v = rng.normal(0, 0.05)
        
        new_u = old_u + delta_u
        new_v = old_v + delta_v
        
        new_u = np.clip(new_u, 0, 1)
        new_v = np.clip(new_v, 0, 1)
        s = new_u + new_v
        if s > 1:
            new_u /= s
            new_v /= s
            
        if new_u == old_u and new_v == old_v:
            continue
            
        t_indices = point_tri_map[k]
        old_areas_subset = areas[t_indices].copy()
        
        current_params[2 * k] = new_u
        current_params[2 * k + 1] = new_v
        
        x_new, y_new = bary_to_cart(new_u, new_v)
        points[k] = [x_new, y_new]
        
        tri_subset = tri_idx[t_indices]
        p1 = points[tri_subset[:, 0]]
        p2 = points[tri_subset[:, 1]]
        p3 = points[tri_subset[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        new_areas_subset = 0.5 * np.abs(cross)
        
        areas[t_indices] = new_areas_subset
        
        new_min_area = np.min(areas)
        
        delta_obj = new_min_area - current_min_area
        
        if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
            current_min_area = new_min_area
        else:
            current_params[2 * k] = old_u
            current_params[2 * k + 1] = old_v
            points[k] = [old_x, old_y]
            areas[t_indices] = old_areas_subset
            
        if current_min_area > best_score:
            best_score = current_min_area
            best_params = current_params.copy()
            
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    best_points = params_to_points(best_params)
    return best_points
