import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Minimum distance between centers of non-overlapping unit hexagons is sqrt(3)
    d_min = np.sqrt(3)
    
    # Valid Initialization
    # 1 center
    # 6 at dist sqrt(3)
    # 4 at dist 3.0, in the gaps (angles 30, 90, 150, 210)
    # This configuration satisfies d_min constraints exactly or better.
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    
    for i in range(6):
        angle = i * np.pi / 3
        pts[i+1] = [d_min * np.cos(angle), d_min * np.sin(angle)]
        
    # Place 4 points at distance 3.0 to ensure they are >= sqrt(3) away from ring 1
    # and >= sqrt(3) away from each other.
    r2 = 3.0
    gap_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6] # 30, 90, 150, 210
    for i in range(4):
        pts[i+7] = [r2 * np.cos(gap_angles[i]), r2 * np.sin(gap_angles[i])]

    # Unit hexagon vertices (side 1, flat top? No, pointy top usually, but symmetry makes it ok if consistent)
    # Let's use pointy-top vertices relative to center:
    # Angles 30, 90, 150, 210, 270, 330 degrees?
    # Actually, a regular hexagon with side 1 has circumradius 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 if one vertex is on x-axis.
    # Let's use vertices at 0, 60, ... 300.
    hex_vertices = np.array([[np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)] for i in range(6)])

    def cost_function(flat_pts):
        p = flat_pts.reshape(N, 2)
        # Max radius (approximate cost for bounding box)
        max_r = np.max(np.linalg.norm(p, axis=1))
        # Min pairwise distance
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        # Penalty for violation
        viol = max(0.0, d_min - min_d)
        # Large penalty for violation to prioritize feasibility
        return max_r + 1000.0 * viol * viol

    # Flatten for scipy
    x0 = pts.flatten()
    
    best_cost = cost_function(x0)
    best_x = x0.copy()
    
    # Use Nelder-Mead for local optimization
    try:
        res = minimize(cost_function, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if res.success or res.fun < best_cost:
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
    except Exception:
        pass
    
    # If time remains, do a second pass with a different initial perturbation
    if time.time() < deadline - 10:
        np.random.seed(42)
        x1 = best_x + np.random.normal(0, 0.1, size=N*2)
        try:
            res2 = minimize(cost_function, x1, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_x = res2.x
        except Exception:
            pass

    best_pts = best_x.reshape(N, 2)
    inner_angles = np.zeros(N) # Assume all inner hexagons are aligned with angle 0

    # Calculate final S (outer side length)
    # We need to find the minimal S for a regular hexagon containing all inner hexagons.
    # The outer hexagon can be rotated.
    # We scan a few candidate outer angles.
    
    def calc_S_for_angle(points, inner_angles, outer_angle_deg):
        theta_out = np.deg2rad(outer_angle_deg)
        # Normals to the sides of the outer hexagon
        # For a hexagon with vertices at 0, 60..., the sides are perpendicular to 30, 90, 150.
        # If rotated by theta_out, normals are at 30+theta_out, 90+theta_out, 150+theta_out.
        n_angles = [np.deg2rad(30) + theta_out, np.deg2rad(90) + theta_out, np.deg2rad(150) + theta_out]
        n_vecs = np.array([[np.cos(na), np.sin(na)] for na in n_angles])
        
        max_proj = 0.0
        for i in range(N):
            c = points[i]
            phi = np.deg2rad(inner_angles[i])
            # Rotate unit hex vertices by phi
            rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            verts = hex_vertices @ rot_mat.T + c
            
            # Project all vertices onto the 3 normal vectors
            # verts shape (6, 2), n_vecs shape (3, 2)
            # proj shape (6, 3)
            projs = np.abs(verts @ n_vecs.T)
            max_proj = max(max_proj, np.max(projs))
            
        # A = max_proj (apothem)
        # S = A / (sqrt(3)/2)
        return 2.0 * max_proj / np.sqrt(3)

    # Scan outer angles
    best_S = float('inf')
    best_outer_angle = 0.0
    
    for ang in [0, 15, 30, 45, 60]:
        S_val = calc_S_for_angle(best_pts, inner_angles, float(ang))
        if S_val < best_S:
            best_S = S_val
            best_outer_angle = ang

    # Format output
    inner = []
    for i in range(N):
        p = best_pts[i]
        inner.append([float(p[0]), float(p[1]), float(inner_angles[i])])
        
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
# EVOLVE-BLOCK-END
