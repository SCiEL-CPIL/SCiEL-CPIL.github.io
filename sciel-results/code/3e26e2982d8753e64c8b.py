<<<<<<< SEARCH
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() > _TIME_DEADLINE:
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

    # Sort candidates by a quick heuristic estimate to process promising ones first
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Use beta=0 as quick estimate
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = pts @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    # Quick sort candidates by estimate
    candidate_estimates = [(axial, quick_estimate(axial)) for axial in candidates]
    candidate_estimates.sort(key=lambda x: x[1])
    
    for axial, est in candidate_estimates:
        if time.time() > _TIME_DEADLINE:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
>>>>>>> REPLACE

