<<<<<<< SEARCH
    # Spacing for non-overlapping unit hexagons
    # Minimum center distance for two unit hexagons (same orientation) = sqrt(3)
    # Use slightly more for numerical safety
    spacing = math.sqrt(3) * 1.001
    
    # Generate compact hex grid positions
    positions = _hex_grid_positions(N, spacing)
    
    # Compute center of mass
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    center = (cx, cy)
    
    # Find minimum enclosing hexagon
    side, angle_deg = _min_enclosing_hex_side(positions, center)
    
    # Add small buffer for numerical safety
    side *= 1.001
=======
    # Spacing for non-overlapping unit hexagons
    # Minimum center distance for two unit hexagons (same orientation) = sqrt(3)
    # Use significantly more for numerical safety and disjointness guarantee
    spacing = math.sqrt(3) * 1.05
    
    # Generate compact hex grid positions
    positions = _hex_grid_positions(N, spacing)
    
    # Compute center of mass
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    center = (cx, cy)
    
    # Find minimum enclosing hexagon
    side, angle_deg = _min_enclosing_hex_side(positions, center)
    
    # Add buffer for numerical safety to ensure outer hexagon contains all inner ones
    side *= 1.05
>>>>>>> REPLACE

