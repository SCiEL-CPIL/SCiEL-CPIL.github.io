import numpy as np

def hexagon_packing_12():
    # Define the 4x3 grid of hexagons
    # 4 rows, 3 columns
    rows = 4
    cols = 3
    
    centers = []
    for i in range(rows):
        for j in range(cols):
            # Lattice coordinates
            # x offset for odd rows
            x = j * np.sqrt(3)
            if i % 2 != 0:
                x += np.sqrt(3) / 2
            y = -i * 1.5
            
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Center the block
    center_of_mass = np.mean(centers, axis=0)
    centers = centers - center_of_mass
    
    # Construct inner hex data
    # Angle is 0
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    # Calculate minimal outer hexagon side length s
    # We need to find the max projection of all vertices onto the 3 axes
    # Axes are at 30, 90, 150 degrees
    
    max_proj = 0.0
    # Generate all vertices of the 12 hexagons
    # Vertex k of hexagon at center (cx, cy) is:
    # (cx + cos(k*pi/3), cy + sin(k*pi/3))
    
    # We can optimize by just checking the support function
    # For a direction u, the max projection of a hexagon centered at c is:
    # c.dot(u) + h(u)
    # where h(u) is the support function of a unit hexagon at origin.
    
    # h(u) for unit hexagon (vertices at 0, 60, ...):
    # h(theta) = cos(delta) where delta is angle diff to nearest vertex.
    # Max h is 1 (at vertices), min is sqrt(3)/2 (at normals).
    
    # Axes:
    axes = []
    for k in range(3):
        angle = np.pi / 6 + k * np.pi / 3 # 30, 90, 150
        axes.append([np.cos(angle), np.sin(angle)])
        
    for u in axes:
        for c in centers:
            proj_center = np.dot(c, u)
            # Calculate h(u)
            # u angle
            u_angle = np.arctan2(u[1], u[0])
            # Find nearest vertex angle (multiple of 60 deg)
            # Vertex angles: 0, pi/3, 2pi/3, ...
            # Normalize u_angle to [0, 2pi)
            u_angle_mod = u_angle % (2 * np.pi)
            # Find closest multiple of pi/3
            k_vertex = round(u_angle_mod / (np.pi / 3))
            vertex_angle = k_vertex * np.pi / 3
            delta = abs(u_angle_mod - vertex_angle)
            # Handle wrap around
            if delta > np.pi:
                delta = 2 * np.pi - delta
            h_u = np.cos(delta)
            
            proj_total = proj_center + h_u
            if abs(proj_total) > max_proj:
                max_proj = abs(proj_total)
                
    # The apothem of the outer hexagon must be >= max_proj
    # Apothem = s * sqrt(3) / 2
    # s = 2 * max_proj / sqrt(3)
    s = 2 * max_proj / np.sqrt(3)
    
    # Outer hex data
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, s
