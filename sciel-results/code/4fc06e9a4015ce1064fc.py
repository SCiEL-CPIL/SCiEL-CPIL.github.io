def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oa_rad = np.radians(outer_angle)
    R = np.array([[np.cos(-oa_rad), -np.sin(-oa_rad)],
                  [np.sin(-oa_rad), np.cos(-oa_rad)]])
    
    # Outer hexagon support function: for each of 6 directions
    outer_normals = hexagon_normals((0, 0), 1.0, 0.0)
    
    max_extent = 0.0
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        rel = inner_verts - np.array(outer_center)
        rel_rot = rel @ R.T  # (6, 2)
        
        for k in range(6):
            d = np.abs(rel_rot @ outer_normals[k])
            d_max = np.max(d)
            if d_max > max_extent:
                max_extent = d_max
    
    return max_extent / (SQRT3 / 2)

def smooth_required_outer_side(centers, outer_center, outer_angle, beta=50.0):
    """Smooth version using log-sum-exp for the max."""
    oa_rad = np.radians(outer_angle)
    R = np.array([[np.cos(-oa_rad), -np.sin(-oa_rad)],
                  [np.sin(-oa_rad), np.cos(-oa_rad)]])
    
    outer_normals = hexagon_normals((0, 0), 1.0, 0.0)
    
    all_extents = []
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        rel = inner_verts - np.array(outer_center)
        rel_rot = rel @ R.T
        
        for k in range(6):
            d = np.abs(rel_rot @ outer_normals[k])
            d_max = np.max(d)
            all_extents.append(d_max)
    
    all_extents = np.array(all_extents)
    # Log-sum-exp for smooth max
    m = np.max(all_extents)
    if m < 1e-12:
        return 0.0
    smooth_max = m + np.log(np.mean(np.exp((all_extents - m) / beta))) * beta
    return smooth_max / (SQRT3 / 2)
