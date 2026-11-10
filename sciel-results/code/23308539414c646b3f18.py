import numpy as np

def hexagon_packing_12():
    # Define the 3x4 grid of hexagons
    v1 = np.array([1.5, np.sqrt(3)/2])
    v2 = np.array([0, np.sqrt(3)])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * v1 + j * v2
            centers.append(c)
            
    centers = np.array(centers)
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Generate all vertices
    angles = np.deg2rad(np.arange(0, 360, 60))
    std_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_vertices = []
    for c in centers:
        verts = c + std_vertices
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Outer hexagon normals
    normal_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    best_s = np.inf
    best_theta = 0
    
    # Fine search
    thetas = np.linspace(0, np.pi/3, 1800)
    
    for theta in thetas:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        rotated_vertices = all_vertices @ R.T
        
        projections = rotated_vertices @ normals.T
        
        max_proj = np.max(projections)
        
        s = (2 / np.sqrt(3)) * max_proj
        
        if s < best_s:
            best_s = s
            best_theta = theta
            
    R_best = np.array([[np.cos(best_theta), -np.sin(best_theta)],
                       [np.sin(best_theta), np.cos(best_theta)]])
    
    final_centers = centers @ R_best.T
    
    inner_angle_deg = np.rad2deg(best_theta)
    
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(final_centers):
        inner_hex_data[i] = [c[0], c[1], inner_angle_deg]
        
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
