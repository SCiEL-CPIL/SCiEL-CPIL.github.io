import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the honeycomb lattice for a 3x4 block
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            centers.append(i * u + j * v)
            
    centers = np.array(centers)
    
    # Vertices of a unit hexagon (side 1) centered at origin
    hex_angles = np.deg2rad(np.arange(6) * 60)
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # All vertices of the 12 hexagons
    all_vertices = []
    for c in centers:
        verts = c + hex_verts
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Function to compute min side length for a given rotation theta
    def min_side_length(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_verts = all_vertices @ rot_matrix.T
        
        # Normals of the outer hexagon
        normal_angles = np.deg2rad(np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projections = rotated_verts @ normals.T
        widths = np.max(projections, axis=0) - np.min(projections, axis=0)
        
        max_width = np.max(widths)
        side_length = max_width / np.sqrt(3)
        return side_length

    # Optimize theta
    best_theta = 0.0
    best_side = float('inf')
    
    # Coarse search
    for theta_deg in np.linspace(0, 60, 601):
        theta = np.deg2rad(theta_deg)
        s = min_side_length(theta)
        if s < best_side:
            best_side = s
            best_theta = theta
            
    # Fine search (Golden section)
    a = np.deg2rad(max(0, np.rad2deg(best_theta) - 2))
    b = np.deg2rad(min(60, np.rad2deg(best_theta) + 2))
    
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = min_side_length(c)
    fd = min_side_length(d)
    
    for _ in range(50):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = min_side_length(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = min_side_length(d)
            
    best_theta = (a + b) / 2
    best_side = min_side_length(best_theta)
    
    # Construct the result
    cos_t = np.cos(best_theta)
    sin_t = np.sin(best_theta)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_verts = all_vertices @ rot_matrix.T
    
    # Calculate optimal translation (center of mass of vertices)
    center_offset = np.mean(rotated_verts, axis=0)
    
    # Final centers
    final_centers = centers @ rot_matrix.T - center_offset
    
    inner = []
    for c in final_centers:
        inner.append([c[0], c[1], np.rad2deg(best_theta)])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
