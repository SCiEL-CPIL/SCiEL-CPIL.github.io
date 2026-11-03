<<<<<<< SEARCH
    # First pass: coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Second pass: fine sweep around the best angle
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
=======
    # First pass: coarse sweep with 30-degree step (exploiting 6-fold symmetry more aggressively)
    angles_coarse = np.arange(0, 60, 3.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Second pass: fine sweep around the best angle
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 3), min(60.0, best_angle + 3), 0.1)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), 0.02)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
>>>>>>> REPLACE

