import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hex_vertices(cx, cy, side, angle_rad):
        # Returns 6x2 array of vertices
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        # verts: 6x2
        # Returns 6x2 array of outward normals
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        # v1, v2: 6x2
        # Returns max overlap depth
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        # Project v1 and v2 onto all normals
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any overlap is <= 0, they are separated
        if np.min(overlaps) <= 1e-9:
            return 0.0
            
        return np.min(overlaps)

    def inside_violation(inner_verts, outer_verts):
        # inner_verts: 6x2, outer_verts: 6x2
        # Returns how much inner_verts stick out of outer_verts
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def objective(params, alpha_rad, theta_rad):
        if time.time() > deadline:
            return 1e10
            
        # params: [x1, y1, ..., x12, y12, s]
        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]
        
        if s < 2.0:
            return 1e6

        # Outer hexagon
        outer_verts = hex_vertices(0, 0, s, theta_rad)
        
        total_penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            
            # Check inside outer
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 100.0  # Higher weight for containment
            
        # Check pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 100.0  # Higher weight for disjointness
                
        return s + total_penalty

    # Initial configuration: Flower
    # 1 center, 6 ring, 5 outer
    # Distances: Center to Ring1: sqrt(3)
    # Ring1 to Ring2: sqrt(3)
    # So Ring2 is at 2*sqrt(3)
    
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    init_pos = []
    # Center
    init_pos.append([0.0, 0.0])
    # Ring 1
    for k in range(6):
        a = np.radians(60 * k)
        init_pos.append([d1 * np.cos(a), d1 * np.sin(a)])
    # Ring 2 (5 hexagons, pick 5 of the 12 spots)
    # Spots at 2*sqrt(3) are at angles 0, 30, 60, ... 330
    # Let's pick 5: 0, 60, 120, 180, 240
    for k in range(5):
        a = np.radians(60 * k)
        init_pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        
    init_pos = np.array(init_pos)
    init_s = 4.5 # Guess
    
    best_s = np.inf
    best_params = None
    best_alpha = 0.0
    best_theta = 0.0
    
    # Grid search over rotations
    # Inner rotation alpha, Outer rotation theta
    # Due to symmetry, we can limit search range.
    # Hexagon has 60 degree rotational symmetry.
    alphas = np.radians([0, 30])
    thetas = np.radians([0, 30])
    
    for alpha in alphas:
        if time.time() > deadline - 20:
            break
        for theta in thetas:
            if time.time() > deadline - 20:
                break
                
            x0 = np.concatenate([init_pos.flatten(), [init_s]])
            
            try:
                res = minimize(
                    objective,
                    x0,
                    args=(alpha, theta),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                # Verify validity before accepting
                pos = res.x[:2 * N].reshape(N, 2)
                s = res.x[2 * N]
                
                # Re-calculate penalty for this solution
                outer_verts = hex_vertices(0, 0, s, theta)
                valid = True
                inner_verts_list = []
                for i in range(N):
                    verts = hex_vertices(pos[i, 0], pos[i, 1], 1.0, alpha)
                    inner_verts_list.append(verts)
                    if inside_violation(verts, outer_verts) > 1e-5:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and s < best_s:
                    best_s = s
                    best_params = res.x.copy()
                    best_alpha = alpha
                    best_theta = theta
                    
            except Exception:
                pass

    # If no valid solution found, try a looser tolerance or different start
    if best_params is None:
        # Fallback: Use a larger s and just return it
        best_s = 5.0
        best_params = np.concatenate([init_pos.flatten(), [best_s]])
        best_alpha = 0
        best_theta = 0

    # Extract final positions
    final_pos = best_params[:2 * N].reshape(N, 2)
    final_s = best_params[2 * N]
    
    # The outer center is (0,0) in our optimization frame.
    # The problem asks for outer center (x,y). We can set it to (0,0).
    # Inner positions are relative to outer center.
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
