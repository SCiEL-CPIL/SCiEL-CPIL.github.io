import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0

    # Hardcoded valid disjoint axial coordinates for 11 hexes (3-5-3)
    axial_coords = [
        (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
    
    # Convert axial to Cartesian for "Pointy Top" hexagons aligned with the grid
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    points = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        points.append([x, y])
    points = np.array(points)
    
    # Get vertices of a single unit hexagon (Pointy Top, centered at origin)
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees
    base_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    base_vertices = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    def calculate_min_side_length(rotation_deg):
        """
        Calculates the minimum side length S of a flat-top outer hexagon
        containing the rotated cluster.
        """
        if time.time() > deadline:
            return float('inf')
            
        rot_rad = np.deg2rad(rotation_deg)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        
        # Rotate all center points
        rotated_points = points @ rot_matrix.T
        
        # Rotate base vertices
        rotated_base_vertices = base_vertices @ rot_matrix.T
        
        # Calculate all vertices of all inner hexagons
        all_vertices = rotated_points[:, np.newaxis, :] + rotated_base_vertices[np.newaxis, :, :]
        all_vertices = all_vertices.reshape(-1, 2)
        
        # Outer Hexagon: Flat-Top
        # Normals at 90, 210, 330 degrees (and opposites)
        normals = np.array([
            [0, 1],
            [-np.sqrt(3)/2, -0.5],
            [np.sqrt(3)/2, -0.5]
        ])
        
        # Project all vertices onto normals
        dists = np.abs(all_vertices @ normals.T)
        
        # Required Apothem A is the max projection
        A = np.max(dists)
        
        # Side length S = 2 * A / sqrt(3)
        # Add a small buffer to ensure strict containment
        S = 2.0 * A / np.sqrt(3) * 1.001
        
        return S

    best_S = float('inf')
    best_rot = 0.0
    
    # Coarse sweep
    angles = np.linspace(0, 60, 361)
    for ang in angles:
        S = calculate_min_side_length(ang)
        if S < best_S:
            best_S = S
            best_rot = ang
            
    # Refine
    if best_S < float('inf'):
        ref_angles = np.linspace(best_rot - 1.0, best_rot + 1.0, 101)
        for ang in ref_angles:
             S = calculate_min_side_length(ang)
             if S < best_S:
                 best_S = S
                 best_rot = ang

    # Construct the result
    rot_rad = np.deg2rad(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    final_centers = points @ rot_matrix.T
    
    inner = []
    for i in range(N):
        x = final_centers[i][0]
        y = final_centers[i][1]
        inner.append([x, y, best_rot])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
