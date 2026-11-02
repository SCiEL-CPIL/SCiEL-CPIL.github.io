<<<<<<< SEARCH
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
=======
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        
        # Try local optimization with different initial angles
        if S < best_global_S and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            angle_deg = math.degrees(beta)
            new_pts, new_angle = local_optimize(pts, angle_deg, max_iter=3000)
            if check_no_overlap(new_pts, new_angle):
                _, new_center, _ = calc_S_for_beta(new_pts, math.radians(new_angle))
                new_S = (float(np.max(np.abs((new_pts - np.array(new_center)) @ np.array([[math.cos(math.radians(new_angle)+math.pi/6+k*math.pi/3), math.sin(math.radians(new_angle)+math.pi/6+k*math.pi/3)] for k in range(6)])).T)) + APOTHEM) / APOTHEM
                if new_S < best_global_S:
                    best_global_S = new_S
                    best_global_pts = new_pts
                    best_global_beta = math.radians(new_angle)
                    best_global_center = new_center
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
>>>>>>> REPLACE

