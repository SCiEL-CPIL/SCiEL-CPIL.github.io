import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _get_all_vertices(inner):
    """
    Get all vertices of all inner hexagons.
    inner: (12, 3) array of (x, y, angle_deg)
    Returns: (72, 2) array of vertices.
    """
    cx = inner[:, 0]
    cy = inner[:, 1]
    angles_deg = inner[:, 2]
    
    base_angles = np.deg2rad(60.0 * np.arange(6))
    
    hex_angles = np.deg2rad(angles_deg)[:, np.newaxis] + base_angles[np.newaxis, :]
    
    cos_a = np.cos(hex_angles)
    sin_a = np.sin(hex_angles)
    
    cx_b = cx[:, np.newaxis]
    cy_b = cy[:, np.newaxis]
    
    verts_x = cx_b + cos_a
    verts_y = cy_b + sin_a
    
    all_verts = np.column_stack((verts_x.ravel(), verts_y.ravel()))
    return all_verts


def _objective(params):
    """
    params: 37 values. 36 for inner (x, y, angle) and 1 for outer angle.
    Returns: s + penalty
    """
    inner_params = params[:36].reshape(12, 3)
    outer_angle_deg = params[36]
    
    all_verts = _get_all_vertices(inner_params)
    
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle_rad + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    proj = all_verts @ normals.T
    
    max_proj = np.max(proj)
    
    s_val = max_proj / (np.sqrt(3.0) / 2.0)
    
    centers = inner_params[:, :2]
    
    min_dist = np.sqrt(3.0)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    mask = np.triu(np.ones((12, 12), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    violations = np.maximum(0, min_dist - pair_dists)
    penalty = np.sum(violations ** 2)
    
    return s_val + penalty * 10.0


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    sqrt3 = np.sqrt(3.0)
    
    x_coords = np.array([-1.5, -0.5, 0.5, 1.5]) * sqrt3
    y_coords = np.array([-1.0, 0.0, 1.0]) * sqrt3
    
    inner = []
    for y in y_coords:
        for x in x_coords:
            inner.append([x, y, 0.0])
            
    inner = np.array(inner)
    outer_angle = 0.0
    
    x0 = np.append(inner.flatten(), outer_angle)
    
    best_s = np.inf
    best_x = x0
    
    # Compute initial s
    all_verts = _get_all_vertices(inner)
    normals = np.column_stack((np.cos(np.deg2rad(30.0 + 60.0 * np.arange(6))), np.sin(np.deg2rad(30.0 + 60.0 * np.arange(6)))))
    proj = all_verts @ normals.T
    s_init = np.max(proj) / (np.sqrt(3.0) / 2.0)
    best_s = s_init
    print(f"Initial s: {best_s:.4f}")
    
    try:
        result = minimize(_objective, x0, method='Powell', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 5000})
        
        if result.fun < best_s:
            inner_res = result.x[:36].reshape(12, 3)
            outer_angle_res = result.x[36]
            
            all_verts = _get_all_vertices(inner_res)
            outer_angle_rad = np.deg2rad(outer_angle_res)
            normal_angles = outer_angle_rad + np.deg2rad(30.0 + 60.0 * np.arange(6))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            proj = all_verts @ normals.T
            s_val = np.max(proj) / (np.sqrt(3.0) / 2.0)
            
            centers = inner_res[:, :2]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            mask = np.triu(np.ones((12, 12), dtype=bool), k=1)
            pair_dists = dists[mask]
            if np.min(pair_dists) >= np.sqrt(3.0) - 1e-6:
                best_s = s_val
                best_x = result.x
                print(f"Optimized s: {best_s:.4f}")
        
        if time.time() < deadline - 60:
            result2 = minimize(_objective, best_x, method='Powell', 
                              options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'maxfev': 10000})
            
            if result2.fun < best_s:
                inner_res = result2.x[:36].reshape(12, 3)
                outer_angle_res = result2.x[36]
                
                all_verts = _get_all_vertices(inner_res)
                outer_angle_rad = np.deg2rad(outer_angle_res)
                normal_angles = outer_angle_rad + np.deg2rad(30.0 + 60.0 * np.arange(6))
                normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
                proj = all_verts @ normals.T
                s_val = np.max(proj) / (np.sqrt(3.0) / 2.0)
                
                centers = inner_res[:, :2]
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                mask = np.triu(np.ones((12, 12), dtype=bool), k=1)
                pair_dists = dists[mask]
                if np.min(pair_dists) >= np.sqrt(3.0) - 1e-6:
                    best_s = s_val
                    best_x = result2.x
                    print(f"Refined s: {best_s:.4f}")
                    
    except Exception as e:
        print(f"Optimization error: {e}")
        
    inner_final = best_x[:36].reshape(12, 3)
    outer_angle_final = best_x[36]
    
    all_verts = _get_all_vertices(inner_final)
    outer_angle_rad = np.deg2rad(outer_angle_final)
    normal_angles = outer_angle_rad + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    proj = all_verts @ normals.T
    s_final = np.max(proj) / (np.sqrt(3.0) / 2.0) + 1e-5
    
    outer = np.array([0.0, 0.0, outer_angle_final])
    
    return inner_final, outer, s_final
