import numpy as np
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 20s margin
    
    # 1. Generate lattice centers for a 3x4 block of hexagons
    # Triangular lattice basis vectors for hexagon centers
    # Distance between centers is sqrt(3)
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 3/2)
    
    # Indices for 4 columns (i=0..3) and 3 rows (j=0..2)
    # Center the lattice
    # i indices: -1.5, -0.5, 0.5, 1.5
    # j indices: -1, 0, 1
    
    i_vals = np.array([-1.5, -0.5, 0.5, 1.5])
    j_vals = np.array([-1.0, 0.0, 1.0])
    
    centers = []
    for i in i_vals:
        for j in j_vals:
            x = i * np.sqrt(3) + j * (np.sqrt(3) / 2)
            y = j * 1.5
            centers.append((x, y))
            
    centers = np.array(centers)
    
    # 2. Generate vertices of the 12 unit hexagons
    # Unit hexagon vertices (orientation 0, vertex at (1,0))
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    all_vertices = []
    for c in centers:
        verts = c + hex_verts
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # 3. Search for optimal rotation of the outer hexagon
    # The inner hexagons are fixed at orientation 0.
    # We rotate the outer hexagon by alpha.
    # This is equivalent to rotating the normal vectors of the outer hexagon.
    
    best_L = float('inf')
    best_alpha = 0.0
    
    # Search range 0 to 60 degrees
    # Step size 0.1 degrees
    num_steps = 600
    alphas = np.linspace(0, np.pi/3, num_steps, endpoint=False)
    
    # Precompute cos/sin for normals?
    # Normals for outer hexagon (flat-top? pointy-top?)
    # Let's define outer hexagon with vertex at angle 0.
    # Normals are at 30, 90, 150, 210, 270, 330.
    # If we rotate outer hexagon by alpha, normals shift by alpha.
    
    base_normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        normal_angles = base_normal_angles + alpha
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Project all vertices onto all normals
        # all_vertices: (72, 2)
        # normals: (6, 2)
        # projections: (72, 6)
        projections = all_vertices @ normals.T
        
        max_proj = np.max(projections)
        
        # L = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        L = 2.0 * max_proj / np.sqrt(3)
        
        if L < best_L:
            best_L = L
            best_alpha = alpha
            
    # 4. Construct the result
    # Outer hexagon: center (0,0), side best_L, rotation best_alpha
    # Inner hexagons: centers as calculated, orientation 0.0
    
    inner = [[float(x), float(y), 0.0] for x, y in centers]
    outer_center = (0.0, 0.0)
    outer_side = float(best_L)
    outer_rotation = float(best_alpha)
    
    return inner, outer_center, outer_side, outer_rotation
