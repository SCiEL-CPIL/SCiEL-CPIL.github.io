import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def compute_min_side(centers, outer_angle_deg):
    sqrt3 = np.sqrt(3)
    a = np.radians(outer_angle_deg)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    all_verts = []
    for cx, cy in centers:
        v = hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    max_width = 0
    for k in range(6):
        proj = all_verts[:, 0] * normals[k, 0] + all_verts[:, 1] * normals[k, 1]
        width = proj.max() - proj.min()
        if width > max_width:
            max_width = width
            
    return max_width / sqrt3

def objective(params, outer_angle_deg):
    centers = params.reshape(N, 2)
    side = compute_min_side(centers, outer_angle_deg)
    penalty = 0.0
    sqrt3 = np.sqrt(3)
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < sqrt3:
                penalty += (sqrt3 - dist)**2
    return side + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    initial_centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        initial_centers.append([x, y])
    initial_centers = np.array(initial_centers)
    initial_centers -= initial_centers.mean(axis=0)
    
    # Fallback result from initial grid
    best_side_fallback = float('inf')
    best_centers_fallback = initial_centers.copy()
    best_outer_center_fallback = [0.0, 0.0]
    best_angle_fallback = 0.0
    
    for angle in [0.0]:
        side = compute_min_side(best_centers_fallback, angle)
        # Refine center
        all_verts = np.vstack([hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in best_centers_fallback])
        a = np.radians(angle)
        normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                             np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
        # Just use mean for fallback
        best_outer_center_fallback = [best_centers_fallback[:, 0].mean(), best_centers_fallback[:, 1].mean()]
        best_side_fallback = side
        best_angle_fallback = angle

    best_side = best_side_fallback
    best_centers = best_centers_fallback.copy()
    best_outer_center = best_outer_center_fallback.copy()
    best_angle = best_angle_fallback
    
    angles_to_try = np.linspace(0, 60, 61)
    
    for angle in angles_to_try:
        if time.time() > deadline:
            break
            
        bounds = [(-10, 10)] * (2 * N)
        
        res = minimize(
            objective, 
            initial_centers.flatten(), 
            args=(angle,), 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )
        
        if res.success:
            centers = res.x.reshape(N, 2)
            
            # Check validity
            valid = True
            sqrt3 = np.sqrt(3)
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < sqrt3 - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side = compute_min_side(centers, angle)
                
                if side < best_side:
                    best_side = side
                    best_centers = centers
                    best_angle = angle
                    
                    # Refine outer center
                    all_verts = np.vstack([hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in best_centers])
                    a = np.radians(angle)
                    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), 
                                         np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
                    
                    def side_for_center(cx, cy, ang):
                        all_v = all_verts
                        a_r = np.radians(ang)
                        n = normals
                        max_w = 0
                        for k in range(6):
                            proj = (all_v[:, 0] - cx) * n[k, 0] + (all_v[:, 1] - cy) * n[k, 1]
                            max_w = max(max_w, proj.max() - proj.min())
                        return max_w / np.sqrt(3)

                    cx, cy = best_centers[:, 0].mean(), best_centers[:, 1].mean()
                    step = 0.1
                    for _ in range(20):
                        improved = False
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                new_side = side_for_center(cx+dx, cy+dy, angle)
                                if new_side < side - 1e-6:
                                    side = new_side
                                    cx += dx
                                    cy += dy
                                    improved = True
                        if not improved:
                            break
                        step *= 0.5
                            
                    best_outer_center = [cx, cy]
                    best_side = side

    inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
    best_side *= 1.001
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_side), float(best_angle)
