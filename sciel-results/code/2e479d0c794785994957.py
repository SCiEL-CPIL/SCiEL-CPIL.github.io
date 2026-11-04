import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices_vectorized(centers, angles):
    """
    Calculate vertices for all hexagons.
    centers: (N, 2)
    angles: (N,) in radians
    Returns: (N, 6, 2)
    """
    # Base vertices of unit hexagon (pointy right, angle 0)
    base_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Rotation matrices
    # angles shape (N,), need (N, 1, 1) for broadcasting with (1, 6, 2)?
    # Let's do it simply:
    # v_rot = R(theta) @ v_base
    # R(theta) = [[cos, -sin], [sin, cos]]
    
    c = np.cos(angles) # (N,)
    s = np.sin(angles) # (N,)
    
    # base_verts is (6, 2)
    # We want to rotate each row of base_verts by the corresponding angle in angles
    # Result should be (N, 6, 2)
    
    # v_x = c * bx - s * by
    # v_y = s * bx + c * by
    
    bx = base_verts[:, 0] # (6,)
    by = base_verts[:, 1] # (6,)
    
    # c is (N, 1), bx is (1, 6) -> (N, 6)
    c_b = c[:, np.newaxis]
    s_b = s[:, np.newaxis]
    
    vx = c_b * bx[np.newaxis, :] - s_b * by[np.newaxis, :]
    vy = s_b * bx[np.newaxis, :] + c_b * by[np.newaxis, :]
    
    verts = np.stack((vx, vy), axis=-1) # (N, 6, 2)
    
    # Translate
    centers_exp = centers[:, np.newaxis, :] # (N, 1, 2)
    verts = verts + centers_exp
    
    return verts

def _min_outer_side_vectorized(centers, inner_angles, outer_angle_rad):
    """
    Calculate minimum side length S of outer hexagon.
    centers: (N, 2)
    inner_angles: (N,) radians
    outer_angle_rad: scalar
    """
    # Normals of outer hexagon
    # Normals are at angles outer_angle + 30 + k*60 degrees
    theta_rad = outer_angle_rad
    normal_angles = theta_rad + np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
    
    # Get all vertices
    verts = _hex_vertices_vectorized(centers, inner_angles) # (N, 6, 2)
    verts_flat = verts.reshape(-1, 2) # (N*6, 2)
    
    # Project all vertices onto all normals
    # dots: (N*6, 6)
    dots = verts_flat @ normals.T
    
    # Max projection
    max_proj = np.max(dots)
    
    # S = max_proj / (sqrt(3)/2)
    S = max_proj / (np.sqrt(3) / 2)
    
    return S

def _overlap_penalty(centers):
    """
    Calculate penalty for overlapping hexagons.
    centers: (N, 2)
    Penalty is sum of squared violations of min distance sqrt(3).
    """
    # Distance matrix
    # d_ij = || c_i - c_j ||
    # We want d_ij >= sqrt(3)
    
    # Vectorized distance calculation
    # centers: (N, 2)
    # diff: (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    # Violation: max(0, sqrt(3) - d)
    min_dist = np.sqrt(3)
    violations = np.maximum(0, min_dist - dists)
    
    # Sum squares
    penalty = np.sum(violations ** 2)
    
    return penalty

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Initial "flower" configuration
    # 1 center at (0,0)
    # 6 neighbors at dist sqrt(3), angles 0, 60, ...
    # 4 outer at dist 3, angles 30, 90, 150, 210 (or similar)
    
    centers = np.zeros((N, 2))
    centers[0] = [0.0, 0.0]
    
    r1 = np.sqrt(3)
    for k in range(1, 7):
        angle = np.deg2rad(k * 60)
        centers[k] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    r2 = 3.0
    # The 4 outer ones in a standard tight packing of 11 are often at the "corners" of the hexagonal cluster
    # Indices 7, 8, 9, 10
    # Let's place them at angles 30, 150, 210, 330? Or 0, 120, 240?
    # Inspiration 1 used: 30, 90, 150, 210 (which is 30 + k*60 for k=0..3? No, 30, 90, 150, 210 is not symmetric. 30, 90, 150, 210, 270, 330 are 6. We need 4.)
    # Let's try angles: 30, 150, 270, 30? No.
    # A common configuration for 11 hexagons is a central hexagon, 6 around it, and 4 more filling the gaps.
    # The gaps are at distance 3 from center, at angles 30, 150, 270, 330?
    # Let's check distance between neighbor at 0 (dist sqrt(3)) and outer at 30 (dist 3).
    # c1 = (sqrt(3), 0). c2 = (3*cos(30), 3*sin(30)) = (3*sqrt(3)/2, 1.5).
    # d^2 = (3*sqrt(3)/2 - sqrt(3))^2 + (1.5 - 0)^2
    # = (sqrt(3)/2)^2 + 1.5^2 = 3/4 + 2.25 = 0.75 + 2.25 = 3.0.
    # d = sqrt(3). Perfect! They touch.
    # So angles 30, 150, 270, 330 work for the 4 outer ones?
    # Let's verify 330. c1 at 300 (or -60): (sqrt(3)/2, -1.5). c2 at 330: (3*sqrt(3)/2, -1.5).
    # d = sqrt(3)/2. Wait.
    # c1 at 0: (sqrt(3), 0).
    # c_outer at 30: (2.598, 1.5).
    # c_outer at 330: (2.598, -1.5).
    # Distance from c1 to c_outer(30) is sqrt(3).
    # Distance from c1 to c_outer(330) is sqrt(3).
    # Distance between c_outer(30) and c_outer(330)?
    # x diff 0, y diff 3. d=3. > sqrt(3). OK.
    # So we can place 4 outer ones at 30, 150, 270, 330.
    
    for k in range(4):
        angle = np.deg2rad(30 + k * 90) # 30, 120, 210, 300? No, 30, 120, 210, 300 is not right.
        # Let's just use 30, 150, 270, 330.
        pass
    
    angles_outer = [30, 150, 270, 330]
    for k, ang in enumerate(angles_outer):
        rad = np.deg2rad(ang)
        centers[7+k] = [r2 * np.cos(rad), r2 * np.sin(rad)]
        
    # Initial inner angles: all 0
    inner_angles = np.zeros(N)
    
    # Initial outer angle: 0
    outer_angle = 0.0
    
    # Pack into a single vector for optimization
    # x = [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, a_out]
    
    def unpack(x):
        centers_opt = np.zeros((N, 2))
        angles_opt = np.zeros(N)
        for i in range(N):
            centers_opt[i, 0] = x[3*i]
            centers_opt[i, 1] = x[3*i+1]
            angles_opt[i] = x[3*i+2]
        outer_angle_opt = x[-1]
        return centers_opt, angles_opt, outer_angle_opt

    def objective(x):
        centers_opt, angles_opt, outer_angle_opt = unpack(x)
        S = _min_outer_side_vectorized(centers_opt, angles_opt, outer_angle_opt)
        P = _overlap_penalty(centers_opt)
        # Penalty weight
        return S + 10.0 * P

    x0 = np.zeros(3 * N + 1)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = inner_angles[i]
    x0[-1] = outer_angle
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, np.pi/3)) # Hexagon symmetry
    bounds.append((0, np.pi/3)) # Outer hexagon symmetry
    
    best_S = float('inf')
    best_x = None
    
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
            
        if i == 0:
            x_start = x0
        else:
            # Random perturbation
            x_start = x0 + np.random.uniform(-0.1, 0.1, len(x0))
            # Ensure angles are in bounds
            for j in range(1, len(x_start), 3):
                 # j is angle index? No, 3*i+2 is angle.
                 pass
            # Simpler: just clip
            x_start[::3] = np.clip(x_start[::3], -10, 10)
            x_start[1::3] = np.clip(x_start[1::3], -10, 10)
            # Angles are at indices 2, 5, 8... and last
            for idx in range(2, len(x_start)-1, 3):
                x_start[idx] = np.clip(x_start[idx], 0, np.pi/3)
            x_start[-1] = np.clip(x_start[-1], 0, np.pi/3)
            
        try:
            res = minimize(objective, x_start, method='Powell', bounds=bounds, 
                           options={'maxiter': 100, 'disp': False})
            
            # Check if valid (low penalty)
            c_opt, a_opt, o_opt = unpack(res.x)
            P = _overlap_penalty(c_opt)
            
            if P < 1e-6: # Valid
                S = _min_outer_side_vectorized(c_opt, a_opt, o_opt)
                if S < best_S:
                    best_S = S
                    best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to initial configuration if optimization failed
        best_x = x0
        best_S = _min_outer_side_vectorized(centers, inner_angles, outer_angle)
        
    # Extract final solution
    centers_final, angles_final, outer_angle_final = unpack(best_x)
    
    # Add tiny margin for safety
    side = best_S + 1e-9
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([centers_final[i, 0], centers_final[i, 1], np.rad2deg(angles_final[i])])
        
    return inner, (0.0, 0.0), side, np.rad2deg(outer_angle_final)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
