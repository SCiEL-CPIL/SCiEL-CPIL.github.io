import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Constants
    EPS = 1e-9
    SQRT3 = np.sqrt(3)
    MIN_DIST = SQRT3 - EPS # Minimum center distance for non-overlap (parallel hexagons)
    
    # Fixed inner hexagons (Center + Ring 1)
    # Center
    fixed_centers = [(0.0, 0.0)]
    # Ring 1: 6 hexagons at distance SQRT3, angles 0, 60, ...
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = SQRT3 * np.cos(angle)
        y = SQRT3 * np.sin(angle)
        fixed_centers.append((x, y))
        
    # Variable inner hexagons (Ring 2)
    # Initial guess: 4 hexagons at distance 3.0, angles 30, 90, 150, 210
    # Note: In a perfect lattice, these would be at distance 3.
    # We will optimize their (x, y) positions.
    init_var_centers = []
    angles2 = [30, 90, 150, 210]
    for ang in angles2:
        angle = np.deg2rad(ang)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        init_var_centers.append([x, y])
        
    # Total centers will be: fixed_centers + var_centers
    # n_fixed = 7, n_var = 4
    
    # Vertices of a unit hexagon (flat-top, rotated 30 deg? No, standard pointy-top is 0, flat-top is 30? 
    # Let's stick to one orientation for all inner hexagons to simplify overlap check.
    # The current program uses angle 30.0 for inner hexagons.
    # Let's define vertices for a hexagon centered at origin, rotated by 30 deg.
    # Standard hexagon vertices (pointy top, side 1):
    # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # Rotated by 30 deg (flat top):
    # (sqrt(3)/2, 0), (0.5, 0.5), (-0.5, 0.5), (-sqrt(3)/2, 0), (-0.5, -0.5), (0.5, -0.5)
    
    # Let's use the vertices from the original code to be consistent, or just define them clearly.
    # Original code:
    # verts_rel = [
    #   [np.sqrt(3)/2, 0.5],
    #   [0, 1],
    #   [-np.sqrt(3)/2, 0.5],
    #   [-np.sqrt(3)/2, -0.5],
    #   [0, -1],
    #   [np.sqrt(3)/2, -0.5]
    # ]
    # This corresponds to a pointy-top hexagon (vertex at (0,1)).
    # The inner hexagons in the original code were listed with angle 30.0 in the `inner` list, 
    # but the `verts_rel` were NOT rotated by 30. 
    # Wait, the `inner` list is just data returned. The `verts_rel` used for checking containment 
    # were static. This implies the inner hexagons are actually pointy-top (angle 0 effectively for geometry), 
    # despite the label 30.0. Or the label is just metadata. 
    # The overlap check for parallel hexagons is valid if they have the same orientation.
    
    verts_rel = np.array([
        [SQRT3/2, 0.5],
        [0, 1],
        [-SQRT3/2, 0.5],
        [-SQRT3/2, -0.5],
        [0, -1],
        [SQRT3/2, -0.5]
    ])
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e9
            
        # Unpack params
        # cx, cy, S, rot
        # x1, y1, x2, y2, x3, y3, x4, y4
        cx, cy, S, rot = params[0], params[1], params[2], params[3]
        var_centers = np.array([
            [params[4], params[5]],
            [params[6], params[7]],
            [params[8], params[9]],
            [params[10], params[11]]
        ])
        
        if S < 0:
            return 1e9
            
        # Combine fixed and variable centers
        all_centers = np.vstack([
            np.array(fixed_centers),
            var_centers
        ])
        
        n_hex = len(all_centers)
        
        # 1. Check Overlaps
        # For each pair, check distance
        min_dist_sq = MIN_DIST * MIN_DIST
        overlap_penalty = 0.0
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                dx = all_centers[i, 0] - all_centers[j, 0]
                dy = all_centers[i, 1] - all_centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    # Penalty proportional to how much they overlap
                    # Use (min_dist - dist)^2
                    overlap_penalty += (MIN_DIST - np.sqrt(dist_sq))**2
                    
        if overlap_penalty > 0.01: # Early exit if huge overlap
             return 1e9 + overlap_penalty

        # 2. Check Containment
        # Generate all vertices
        all_verts = np.empty((n_hex * 6, 2))
        for i, (cx_i, cy_i) in enumerate(all_centers):
            all_verts[i*6:(i+1)*6] = verts_rel + [cx_i, cy_i]
            
        # Transform to outer hex frame
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        
        c_rot = np.cos(-rot)
        s_rot = np.sin(-rot)
        lx = dx * c_rot - dy * s_rot
        ly = dx * s_rot + dy * c_rot
        
        c = 2.0 / SQRT3
        
        # Inequalities for point (lx, ly) in hex centered at (0,0) side S (pointy-top)
        v1 = ly - S
        v2 = -ly - S
        v3 = c * lx - ly - S
        v4 = -c * lx - ly - S
        v5 = c * lx + ly - S
        v6 = -c * lx + ly - S
        
        violations = np.maximum(0, np.maximum(np.maximum(v1, v2), np.maximum(v3, np.maximum(v4, np.maximum(v5, v6)))))
        
        containment_penalty = np.sum(violations**2)
        
        return S + 100 * (containment_penalty + overlap_penalty)

    # Initial guess
    # Outer center: centroid of all centers
    all_init_centers = np.vstack([
        np.array(fixed_centers),
        np.array(init_var_centers)
    ])
    cx0, cy0 = np.mean(all_init_centers, axis=0)
    
    x0 = np.array([cx0, cy0, 3.5, 0.0] + [coord for c in init_var_centers for coord in c])
    
    # Optimize
    # Use Nelder-Mead
    # 12 variables
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000, 'adaptive': True})
    
    if time.time() - start_time > DEADLINE:
        # Return best found or initial?
        # If timeout, return a valid static configuration
        inner_static = []
        for x, y in fixed_centers:
            inner_static.append([x, y, 30.0])
        for x, y in init_var_centers:
            inner_static.append([x, y, 30.0])
        return inner_static, (0.0, 0.0), 3.5, 0.0
        
    cx_opt, cy_opt, S_opt, rot_opt = res.x[0], res.x[1], res.x[2], res.x[3]
    var_centers_opt = np.array([
        [res.x[4], res.x[5]],
        [res.x[6], res.x[7]],
        [res.x[8], res.x[9]],
        [res.x[10], res.x[11]]
    ])
    
    S_opt += 1e-6 # Safety margin
    
    # Construct final inner list
    inner = []
    for x, y in fixed_centers:
        inner.append([x, y, 30.0])
    for x, y in var_centers_opt:
        inner.append([x, y, 30.0])
        
    return inner, (cx_opt, cy_opt), S_opt, np.rad2deg(rot_opt)
