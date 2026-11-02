<<<<<<< SEARCH
    # Try all combinations of 4 from ring2
    # C(14, 4) = 1001 combinations - very fast
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Try several outer hexagon configurations
        candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (cx_mean, cy_mean, 15.0),
            (cx_mean, cy_mean, 45.0),
        ]
        
        for ocx, ocy, oang in candidates:
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            # First get a quick estimate
            side_est = _min_outer_side(inner_hexes, ocx, ocy, oang)
            if side_est >= best_side * 1.001:
                continue  # skip if clearly worse
            # Local optimization
            ocx2, ocy2, oang2, side = _optimize_outer(inner_hexes, ocx, ocy, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (ocx2, ocy2)
                best_angle = oang2
=======
    # Try all combinations of 4 from ring2
    # C(13, 4) = 715 combinations
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Try several outer hexagon configurations with more angle diversity
        candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (cx_mean, cy_mean, 15.0),
            (cx_mean, cy_mean, 45.0),
            (cx_mean, cy_mean, 60.0),
            (cx_mean, cy_mean, -15.0),
            (cx_mean, cy_mean, -30.0),
            (cx_mean, cy_mean, 75.0),
        ]
        
        for ocx, ocy, oang in candidates:
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            # First get a quick estimate
            side_est = _min_outer_side(inner_hexes, ocx, ocy, oang)
            if side_est >= best_side * 1.0005:
                continue  # skip if clearly worse
            # Check time budget before expensive optimization
            if time.time() > deadline - 5.0:
                break
            # Local optimization
            ocx2, ocy2, oang2, side = _optimize_outer(inner_hexes, ocx, ocy, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (ocx2, ocy2)
                best_angle = oang2
>>>>>>> REPLACE

