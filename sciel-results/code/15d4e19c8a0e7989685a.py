<<<<<<< SEARCH
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_idx = -1

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            best_global_idx = idx
=======
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_idx = -1

    # First pass: evaluate all candidates
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts.copy()
            best_global_beta = beta
            best_global_center = center
            best_global_idx = idx

    # Second pass: local search on the best configuration's point positions
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 20:
        try:
            def full_objective_v2(pts_local, cx, cy, beta):
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = np.array([
                    [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                     math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                    for k in range(6)
                ])
                
                pts_centered = pts_local - np.array([cx, cy])
                projs = pts_centered @ normals.T
                
                # For each normal, we need max of |dot|
                max_proj = 0.0
                for k in range(6):
                    col = projs[:, k]
                    max_proj = max(max_proj, float(np.max(col)), float(-np.min(col)))
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            
            # Try perturbing individual hexagons
            n_pts = len(best_global_pts)
            improved = True
            max_iter = 5
            iter_count = 0
            
            while improved and iter_count < max_iter and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 10:
                improved = False
                iter_count += 1
                for i in range(n_pts):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 5:
                        break
                    # Try small perturbations in multiple directions
                    for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                        for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                            if dx == 0.0 and dy == 0.0:
                                continue
                            pts_perturbed = best_global_pts.copy()
                            pts_perturbed[i][0] += dx
                            pts_perturbed[i][1] += dy
                            
                            # Optimize outer hexagon for perturbed points
                            S_pert, beta_pert, center_pert = optimize_shape(pts_perturbed)
                            if S_pert < best_global_S - 1e-8:
                                best_global_S = S_pert
                                best_global_pts = pts_perturbed
                                best_global_beta = beta_pert
                                best_global_center = center_pert
                                improved = True
                                break
                        if improved:
                            break
        except Exception:
            pass
>>>>>>> REPLACE

