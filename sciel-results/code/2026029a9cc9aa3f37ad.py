    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        # Scale points to ensure strict disjointness
        centroid = np.mean(pts, axis=0)
        pts_scaled_inner = centroid + (pts - centroid) * 1.001 # 0.1% expansion
        S, beta, center = optimize_shape(pts_scaled_inner)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_scaled_inner # Store scaled points
            best_global_beta = beta
            best_global_center = center
