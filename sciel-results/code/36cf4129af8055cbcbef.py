<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate initializations
    initializations = []
    
    # BCC lattice with DE optimization (primary)
    if time.time() < deadline - 60.0:
        V_bcc = _generate_bcc_points(4)
        params = _lattice_optimize_bcc(V_bcc, time_limit=25.0)
        
        if params is not None:
            s, tx, ty, tz, rx, ry, rz = params
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            pts_bcc = s * (V_bcc @ R.T) + np.array([tx, ty, tz])
            pts_bcc = np.clip(pts_bcc, -0.49, 0.49)
            initializations.append(pts_bcc)
        
        # Also add unoptimized BCC
        V_bcc_simple = _generate_bcc_points(4)
        V_bcc_simple -= V_bcc_simple.mean(axis=0)
        scale = 0.5 / np.max(np.abs(V_bcc_simple)) * 0.9
        pts_bcc_simple = scale * V_bcc_simple
        pts_bcc_simple = np.clip(pts_bcc_simple, -0.49, 0.49)
        initializations.append(pts_bcc_simple)
    
    # FCC lattice
    fcc_centers = _generate_fcc()
    initializations.append(fcc_centers)
    
    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)
    
    # Perturbed versions
    for i in range(4):
        if time.time() > deadline - 50.0:
            break
        base = _generate_bcc_points(4)
        base -= base.mean(axis=0)
        scale = 0.5 / np.max(np.abs(base)) * 0.9
        base = scale * base
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    for i in range(3):
        if time.time() > deadline - 45.0:
            break
        base = _generate_fcc()
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    # Random
    for i in range(2):
        if time.time() > deadline - 40.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)
    
    # Phase 2: Optimize each initialization with multiple target radii
    target_radii = [0.105, 0.108, 0.110, 0.111, 0.112, 0.113]
    
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 35.0:
            break
        
        # Try each target radius from this initialization
        for target_r in target_radii:
            if time.time() > deadline - 20.0:
                break
            
            # Add small noise to break symmetry
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)
            
            opt_pts = _local_optimize_violation(init_pts, target_r, deadline)
            
            if np.all(np.isfinite(opt_pts)):
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()
    
    # Phase 3: SGLD refinement on best solution
    if best_centers is not None and time.time() < deadline - 20.0:
        target_r = best_radius + 1e-5
        sgld_pts = _sgld_refine(best_centers, target_r, deadline, n_steps=3000)
        r_sgld = _compute_radius(sgld_pts)
        if r_sgld > best_radius:
            best_radius = r_sgld
            best_centers = sgld_pts
    
    # Phase 4: High-precision polish
    if best_centers is not None and time.time() < deadline - 15.0:
        best_centers, best_radius = _high_precision_polish(best_centers, deadline)
    
    # Phase 5: Final contact graph refinement
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(2):
            if time.time() > deadline - 3.0:
                break
            best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)
    
    # Final validation
    if best_centers is None:
        best_centers = _generate_grid()
    
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)
    
    return best_centers
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # BCC lattice (scaled to fit)
    V_bcc = _generate_bcc_points(4)
    V_bcc -= V_bcc.mean(axis=0)
    scale_bcc = 0.45 / np.max(np.abs(V_bcc))
    pts_bcc = scale_bcc * V_bcc
    initializations.append(pts_bcc.copy())
    
    # FCC lattice
    initializations.append(_generate_fcc())
    
    # Grid
    initializations.append(_generate_grid())
    
    # Random initializations with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        initializations.append(_generate_random(seed))
    
    # Perturbed BCC
    for i in range(3):
        noise_scale = 0.01 + 0.01 * i
        perturbed = pts_bcc + rng.normal(0, noise_scale, size=(N, 3))
        initializations.append(np.clip(perturbed, -0.48, 0.48))
    
    # Phase 2: Riesz energy optimization from each initialization
    # Use multiple s values for diversity
    s_values = [3.5, 4.0, 5.0]
    
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
        
        for s in s_values:
            if time.time() > deadline - 25.0:
                break
            
            # Add tiny noise to break exact symmetry
            noise = rng.normal(0, 0.001, size=(N, 3))
            init_pts = np.clip(init + noise, -0.48, 0.48)
            
            opt_pts = _riesz_optimize(init_pts, deadline, s=s, wall_weight=10.0, max_iter=1500)
            
            if np.all(np.isfinite(opt_pts)):
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()
    
    # Phase 3: SGLD refinement on best solution
    if best_centers is not None and time.time() < deadline - 25.0:
        target_r = best_radius + 1e-4
        sgld_pts = _sgld_refine(best_centers, target_r, deadline, n_steps=2000)
        r_sgld = _compute_radius(sgld_pts)
        if r_sgld > best_radius:
            best_radius = r_sgld
            best_centers = sgld_pts
    
    # Phase 4: High-precision polish
    if best_centers is not None and time.time() < deadline - 15.0:
        best_centers, best_radius = _high_precision_polish(best_centers, deadline)
    
    # Phase 5: Final contact graph refinement
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(2):
            if time.time() > deadline - 3.0:
                break
            best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)
    
    # Final validation
    if best_centers is None:
        best_centers = _generate_grid()
    
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)
    
    return best_centers
>>>>>>> REPLACE
