import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    thetas = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _min_enclosing_hex_side_fast(centers, angles, outer_angle_deg, outer_center):
    """Compute the minimum enclosing hexagon side length."""
    n = len(centers)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = np.deg2rad(angles[i])
        thetas = a + np.arange(6) * np.pi / 3
        all_verts[i*6:(i+1)*6, 0] = centers[i][0] + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = centers[i][1] + np.sin(thetas)
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = all_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Rigid flower packing: 1 center + 6 first ring + 4 second ring
    # All hexagons have the same orientation (30 degrees)
    
    angles = np.array([30.0] * 11)
    
    centers_list = [[0.0, 0.0]]
    
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Second ring: 4 hexagons at distance 3
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    
    centers = np.array(centers_list)
    n_inner = len(centers)
    
    # Precompute all vertices
    all_verts = np.zeros((n_inner * 6, 2))
    for i in range(n_inner):
        a = np.deg2rad(angles[i])
        thetas = a + np.arange(6) * np.pi / 3
        all_verts[i*6:(i+1)*6, 0] = centers[i][0] + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = centers[i][1] + np.sin(thetas)
    
    # Search over outer hexagon rotation angles in [0, 60)
    best_S = float('inf')
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # Coarse grid: 0.5 degree steps
    angle_candidates = np.linspace(0, 60, 121)
    
    for ang in angle_candidates:
        if time.time() > DEADLINE - 10.0:
            break
        
        normals = _get_outer_normals(ang)
        
        # Find optimal center: minimize max |v·n - c·n| over all v and n
        # This is a linear programming problem. Use Nelder-Mead on 2D center.
        
        def obj_center(c, normals=normals, all_verts=all_verts):
            dots = (all_verts - c) @ normals.T
            return np.max(np.abs(dots))
        
        c0 = np.mean(all_verts, axis=0)
        
        res = minimize(obj_center, c0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        
        max_proj = res.fun
        S = 2 * max_proj / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_outer_center = res.x
            best_outer_angle = ang
    
    # Fine-tune around best angle
    if time.time() < DEADLINE - 5.0:
        for ang in np.linspace(best_outer_angle - 0.5, best_outer_angle + 0.5, 101):
            if time.time() > DEADLINE - 3.0:
                break
            
            normals = _get_outer_normals(ang)
            
            def obj_center(c, normals=normals, all_verts=all_verts):
                dots = (all_verts - c) @ normals.T
                return np.max(np.abs(dots))
            
            c0 = best_outer_center
            res = minimize(obj_center, c0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            S = 2 * res.fun / np.sqrt(3)
            
            if S < best_S:
                best_S = S
                best_outer_center = res.x
                best_outer_angle = ang
    
    # Final polish
    if time.time() < DEADLINE - 2.0:
        normals = _get_outer_normals(best_outer_angle)
        
        def obj_center(c, normals=normals, all_verts=all_verts):
            dots = (all_verts - c) @ normals.T
            return np.max(np.abs(dots))
        
        res = minimize(obj_center, best_outer_center, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
        
        if res.fun * 2 / np.sqrt(3) < best_S:
            best_S = 2 * res.fun / np.sqrt(3)
            best_outer_center = res.x
    
    # Build result
    inner = []
    for i in range(n_inner):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    outer_center = (best_outer_center[0], best_outer_center[1])
    side = best_S
    outer_angle = best_outer_angle
    
    return inner, outer_center, side, outer_angle
