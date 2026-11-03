import numpy as np
import time

N = 11

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    # 1. Define the fixed rigid configuration of 11 hexagons
    # Axial coordinates (q, r) for a compact 11-hexagon cluster
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 (partial): (2,-1), (2,-2), (-2,1), (-2,2)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (-2, 1), (-2, 2)
    ]
    
    sqrt3 = np.sqrt(3)
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster at origin
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass

    # 2. Precompute vertices for each hexagon in local frame (rotation 0)
    # Unit hexagon vertices at angle 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    local_verts = []
    for i in range(6):
        theta = i * np.pi / 3.0
        local_verts.append((np.cos(theta), np.sin(theta)))
    local_verts = np.array(local_verts)
    
    # All local vertices for all 11 hexagons (relative to their centers)
    # Shape: (11, 6, 2)
    all_local_verts = np.repeat(local_verts[np.newaxis, :, :], N, axis=0)
    
    # 3. Function to calculate minimum enclosing hexagon side length
    # for a given cluster rotation theta_c
    # Outer hexagon is fixed at angle 0.
    
    def min_side_length(theta_c):
        # Rotate all centers and local vertices by theta_c
        cos_c = np.cos(theta_c)
        sin_c = np.sin(theta_c)
        rot_mat = np.array([[cos_c, -sin_c], [sin_c, cos_c]])
        
        # New centers
        rotated_centers = centers @ rot_mat.T
        
        # New local vertices (rotated)
        rotated_local_verts = np.einsum('ij,kjl->kil', rot_mat, all_local_verts)
        
        # Absolute vertices = rotated_centers + rotated_local_verts
        # Shape: (11, 6, 2)
        abs_verts = rotated_local_verts + rotated_centers[:, np.newaxis, :]
        
        # Flatten to (66, 2)
        abs_verts_flat = abs_verts.reshape(-1, 2)
        
        # Outer hexagon: Center (0,0), Angle 0, Side S
        # Outward normals for hexagon with vertex at angle 0:
        # Normals at 30, 90, 150, 210, 270, 330.
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto normals
        # dots: (66, 6)
        dots = abs_verts_flat @ normals.T
        
        # Max absolute projection for each normal
        max_projections = np.max(np.abs(dots), axis=0)
        
        # Required apothem A = max(max_projections)
        A = np.max(max_projections)
        
        # Side S = 2 * A / sqrt(3)
        S = 2.0 * A / sqrt3
        
        return S

    # 4. Optimize theta_c
    # The function is periodic with period 60 deg (pi/3) due to hexagonal symmetry.
    # We can scan 0 to pi/3.
    
    best_S = np.inf
    best_theta_c = 0.0
    
    # Coarse scan
    n_scan = 100
    thetas = np.linspace(0, np.pi/3, n_scan)
    for theta in thetas:
        S = min_side_length(theta)
        if S < best_S:
            best_S = S
            best_theta_c = theta
            
    # Refine around best
    if time.time() - t0 < DEADLINE - 10:
        step = (np.pi/3) / n_scan
        for _ in range(5):
            local_thetas = np.linspace(best_theta_c - step, best_theta_c + step, 20)
            local_best_S = np.inf
            local_best_theta = best_theta_c
            for theta in local_thetas:
                S = min_side_length(theta)
                if S < local_best_S:
                    local_best_S = S
                    local_best_theta = theta
            if local_best_S >= best_S:
                break
            best_S = local_best_S
            best_theta_c = local_best_theta
            step /= 5.0

    # 5. Construct final output
    # We fixed outer hexagon angle to 0.
    # We need to output the inner hexagons rotated by best_theta_c.
    # Inner hexagons: (x, y, angle_degrees)
    # The angle of each inner hexagon is best_theta_c (since they are all aligned in the cluster).
    
    cos_c = np.cos(best_theta_c)
    sin_c = np.sin(best_theta_c)
    rot_mat = np.array([[cos_c, -sin_c], [sin_c, cos_c]])
    
    final_centers = centers @ rot_mat.T
    
    inner = []
    angle_deg = np.rad2deg(best_theta_c)
    for i in range(N):
        x = final_centers[i, 0]
        y = final_centers[i, 1]
        inner.append([x, y, angle_deg])
    
    outer_center = (0.0, 0.0)
    outer_side = best_S * 1.0001 # Safety margin
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
