# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _get_normals(a_deg):
    a = np.deg2rad(a_deg)
    normals = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(normals), np.sin(normals)])

def _overlap_depth(v1, v2, n1, n2):
    all_normals = np.vstack([n1, n2])
    min_gap = np.inf
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        gap = min(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
        if gap < min_gap:
            min_gap = gap
    return -min_gap if min_gap < 0 else 0.0

def _compute_objective(params):
    centers = params[:22].reshape(11, 2)
    angles = params[22:]
    
    vertices_list = []
    normals_list = []
    for i in range(11):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        n = _get_normals(angles[i])
        vertices_list.append(v)
        normals_list.append(n)
        
    outer_normals = _get_normals(0.0)
    max_proj = 0.0
    for v in vertices_list:
        projs = v @ outer_normals.T
        abs_projs = np.abs(projs)
        max_proj = max(max_proj, np.max(abs_projs))
        
    side = max_proj / (np.sqrt(3.0) / 2.0)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            depth = _overlap_depth(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j])
            if depth > 0:
                penalty += depth ** 2
                
    return side + 100.0 * penalty

def hexagon_packing_11():
    t_start = time.time()
    deadline = t_start + 580.0
    
    dx = 2.5
    dy = 3.0 * np.sqrt(3.0) / 2.0
    initial_centers = np.array([
        [0.0, 0.0],
        [-dx, 0.0], [dx, 0.0],
        [-dx/2, dy], [dx/2, dy],
        [-dx/2, -dy], [dx/2, -dy],
        [-1.5*dx, dy], [1.5*dx, dy],
        [-1.5*dx, -dy], [1.5*dx, -dy],
    ])
    
    x0 = np.hstack([initial_centers.flatten(), np.zeros(11)])
    
    best_val = np.inf
    best_x = x0.copy()
    
    def local_min(x):
        res = minimize(
            _compute_objective, x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return res.x, res.fun, 0

    try:
        bh = basinhopping(
            _compute_objective, x0,
            minimizer_kwargs={'method': 'Nelder-Mead'},
            niter=50,
            stepsize=0.5,
            minimizer=local_min,
            seed=42
        )
        
        if bh.fun < best_val:
            best_val = bh.fun
            best_x = bh.x
            
    except Exception:
        pass

    # Final refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _compute_objective, best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    centers = best_x[:22].reshape(11, 2)
    angles = best_x[22:]
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    vertices_list = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(11)]
    all_vertices = np.vstack(vertices_list)
    outer_normals = _get_normals(0.0)
    max_proj = np.max(np.abs(all_vertices @ outer_normals.T))
    side = max_proj / (np.sqrt(3.0) / 2.0)
    
    return inner_hex_data, outer_hex_data, float(side)
# EVOLVE-BLOCK-END
