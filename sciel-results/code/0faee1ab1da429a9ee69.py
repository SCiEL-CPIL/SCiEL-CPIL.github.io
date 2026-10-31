def _min_required_side(positions, angles, outer_angle=0.0):
    max_val = 0.0
    outer_a = np.radians(outer_angle)
    thetas = outer_a + np.pi/6 + np.arange(6) * np.pi / 3
    
    # Collect all vertices
    all_verts = []
    for i, (px, py) in enumerate(positions):
        verts = _hex_vertices(px, py, 1.0, angles[i])
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0
        
    all_verts = np.array(all_verts)
    
    # Center the points at the origin
    centroid = np.mean(all_verts, axis=0)
    centered_verts = all_verts - centroid
    
    for vx, vy in centered_verts:
        for theta in thetas:
            val = vx * np.cos(theta) + vy * np.sin(theta)
            if val > max_val:
                max_val = val
                
    return max_val / (np.sqrt(3) / 2)
