def calculate_outer_side(verts_list):
    all_points = np.vstack(verts_list)
    cx = np.mean(all_points[:, 0])
    cy = np.mean(all_points[:, 1])
    pts = all_points - np.array([cx, cy])
    
    min_side = float('inf')
    best_angle = 0.0
    
    for outer_angle_deg in range(0, 30, 5):
        outer_angle_rad = np.radians(outer_angle_deg)
        side = required_outer_side(pts, 0.0, 0.0, outer_angle_rad)
        if side < min_side:
            min_side = side
            best_angle = outer_angle_deg
            
    return min_side, cx, cy, best_angle
