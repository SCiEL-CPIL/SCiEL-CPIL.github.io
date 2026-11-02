import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for hexagonal packing of unit hexagons
    # Distance between centers of touching hexagons is sqrt(3)
    # Axial coordinates (q, r) to Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Generate all hexagons in a side-3 hexagon (19 hexes)
    # |q| <= 2, |r| <= 2, |q+r| <= 2
    all_hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                all_hexes.append((q, r))
                
    # Identify the core 7 hexes (side-2 hexagon)
    # |q| <= 1, |r| <= 1, |q+r| <= 1
    core_hexes = []
    outer_hexes = []
    for q, r in all_hexes:
        if abs(q) <= 1 and abs(r) <= 1 and abs(q + r) <= 1:
            core_hexes.append((q, r))
        else:
            outer_hexes.append((q, r))
            
    # core_hexes should have 7 elements
    # outer_hexes should have 12 elements
    
    best_S = float('inf')
    best_theta = 0.0
    best_cluster_indices = None
    
    # Pre-compute Cartesian centers for all hexes
    centers = {hr: axial_to_cartesian(hr[0], hr[1]) for hr in all_hexes}
    
    # Function to calculate min side length for a given cluster and rotation
    def calc_min_side(cluster_axial, theta):
        # Get centers
        pts = np.array([centers[hr] for hr in cluster_axial])
        
        # Generate vertices for each hexagon
        # Hexagon side 1, angle 0 (flat top? pointy top?)
        # Standard hexagon vertices at angles 0, 60, ...
        # If angle 0, vertices are at (1,0), (0.5, sqrt(3)/2), ...
        # This corresponds to "pointy" top/bottom if y is up?
        # cos(0)=1, sin(0)=0 -> (1,0)
        # cos(60)=0.5, sin(60)=0.866
        # This is a hexagon with a vertex on the x-axis.
        
        angles = np.arange(0, 2 * np.pi, np.pi / 3)
        base_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # All vertices for all hexes
        # pts shape (12, 2), base_vertices shape (6, 2)
        # vertices = pts[:, np.newaxis, :] + base_vertices[np.newaxis, :, :]
        vertices = pts[:, np.newaxis, :] + base_vertices[np.newaxis, :, :]
        
        # Rotate vertices by theta
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        vertices = vertices @ R.T
        
        # Outer hexagon normals
        # Outer hexagon is fixed at angle 0.
        # Normals at 0, 60, 120, 180, 240, 300 degrees
        normal_angles = np.arange(0, 2 * np.pi, np.pi / 3)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project vertices onto normals
        # vertices shape (12, 6, 2), normals shape (6, 2)
        # We want max dot product for each normal
        # dots[k, i, j] = vertices[i, j, :] dot normals[k, :]
        
        # Efficient calculation:
        # max_proj[k] = max_{i,j} (v_{i,j} . n_k)
        # = max_{i,j} (sum_m v_{i,j,m} n_{k,m})
        
        # Reshape vertices to (72, 2)
        v_flat = vertices.reshape(-1, 2)
        
        # dots shape (6, 72)
        dots = normals @ v_flat.T
        
        max_dots = np.max(dots, axis=1)
        
        # S = 2/sqrt(3) * max(max_dots)
        return (2.0 / sqrt3) * np.max(max_dots)

    # Search over all combinations of 5 outer hexes
    # To speed up, we can sample or just do all 792
    for combo in combinations(outer_hexes, 5):
        cluster = core_hexes + list(combo)
        
        # Optimize theta
        res = minimize_scalar(
            lambda th: calc_min_side(cluster, th),
            bounds=(0, np.pi / 3),
            method='bounded'
        )
        
        if res.fun < best_S:
            best_S = res.fun
            best_theta = res.x
            best_cluster_indices = cluster

    # Construct the final result
    final_cluster = best_cluster_indices
    final_theta = best_theta
    
    # Calculate final positions and angles
    inner = []
    for q, r in final_cluster:
        c = centers[(q, r)]
        # Rotate center
        c_rot = c @ np.array([[np.cos(final_theta), -np.sin(final_theta)], 
                              [np.sin(final_theta), np.cos(final_theta)]])
        # Angle in degrees
        angle_deg = np.degrees(final_theta)
        inner.append([float(c_rot[0]), float(c_rot[1]), float(angle_deg)])
        
    # Outer hexagon
    # Center (0,0), Side best_S, Angle 0
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
