<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Differential Evolution global search
    # Use a smooth objective for DE (it needs gradients implicitly via finite diff)
    # The delaunay_smooth_obj is smoother than hard min
    
    def de_objective(x_flat):
        if time.time() > deadline - 10:
            return 1e10  # Penalty if we're running out of time
        bary = x_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        # Use a moderately smooth objective
        return _delaunay_smooth_obj(bary.flatten(), gamma=30.0)
    
    # Bounds for DE: u_i in [0, 1], v_i in [0, 1]
    de_bounds = [(0, 1)] * (N * 2)
    
    # Generate multiple diverse starting points for DE
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    # Run DE with moderate population
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.success:
                de_bary = de_result.x.reshape(N, 2)
                update_best(de_bary)
    except Exception:
        pass
    
    # Phase 2: Local polish from multiple diverse starts
    # Use the best found and several random/perturbed starts
    polish_starts = []
    
    if best_bary is not None:
        polish_starts.append(best_bary.copy())
    
    # Add hex init
    polish_starts.append(hex_init.copy())
    
    # Add some random starts
    for i in range(8):
        if time.time() > deadline - 20:
            break
        rand = generate_random_init(seed=500 + i)
        polish_starts.append(rand)
    
    # Add perturbed versions of best
    if best_bary is not None:
        for i in range(4):
            if time.time() > deadline - 15:
                break
            pert = generate_perturbed_init(best_bary, scale=0.03, seed=600 + i)
            polish_starts.append(pert)
    
    # Polish each start
    for start_bary in polish_starts:
        if time.time() > deadline - 10:
            break
        
        # First pass: moderate beta
        polished = _local_polish_bary(start_bary, beta=300.0, maxiter=200)
        update_best(polished)
        
        # Second pass: higher beta for tighter min
        if time.time() > deadline - 8:
            break
        polished2 = _local_polish_bary(polished, beta=800.0, maxiter=300)
        update_best(polished2)
    
    # Phase 3: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        final = _local_polish_bary(best_bary, beta=1500.0, maxiter=500)
        update_best(final)
        
        # One more with even higher beta
        if time.time() < deadline - 3:
            final2 = _local_polish_bary(final, beta=3000.0, maxiter=300)
            update_best(final2)
    
    # Convert best barycentric to Cartesian and return
    if best_bary is not None:
        return bary_to_cart(best_bary)
    else:
        # Fallback: return random valid points
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        return bary_to_cart(bary)
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Generate diverse initial configurations
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    grid_init = generate_grid_init()
    update_best(grid_init)
    
    # Run DE global search
    def de_objective(x_flat):
        if time.time() > deadline - 15:
            return 1e10
        bary = x_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        return _delaunay_smooth_obj(bary.flatten(), gamma=30.0)
    
    de_bounds = [(0, 1)] * (N * 2)
    
    try:
        if time.time() < deadline - 50:
            de_result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.success:
                de_bary = de_result.x.reshape(N, 2)
                update_best(de_bary)
    except Exception:
        pass
    
    # Phase 2: Multi-method local polish
    polish_starts = []
    
    if best_bary is not None:
        polish_starts.append(best_bary.copy())
    polish_starts.append(hex_init.copy())
    polish_starts.append(grid_init.copy())
    
    # Add random starts
    for i in range(6):
        if time.time() > deadline - 25:
            break
        rand = generate_random_init(seed=500 + i)
        polish_starts.append(rand)
    
    # Add perturbed versions
    if best_bary is not None:
        for i in range(3):
            if time.time() > deadline - 20:
                break
            pert = generate_perturbed_init(best_bary, scale=0.03, seed=600 + i)
            polish_starts.append(pert)
    
    # Polish each start with both L-BFGS-B (soft min) and SLSQP (constrained)
    for start_bary in polish_starts:
        if time.time() > deadline - 12:
            break
        
        # Method 1: L-BFGS-B with soft-min (increasing beta)
        polished = _local_polish_bary(start_bary, beta=300.0, maxiter=150)
        update_best(polished)
        
        if time.time() > deadline - 10:
            break
        polished2 = _local_polish_bary(polished, beta=800.0, maxiter=200)
        update_best(polished2)
        
        # Method 2: SLSQP with area constraints
        if time.time() > deadline - 8:
            break
        current_best_area = best_val if best_val > 0 else 0.01
        slsqp_result = _slsqp_polish(polished2, current_best_area, maxiter=100)
        update_best(slsqp_result)
    
    # Phase 3: Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        final = _local_polish_bary(best_bary, beta=1500.0, maxiter=300)
        update_best(final)
        
        if time.time() < deadline - 3:
            final2 = _local_polish_bary(final, beta=3000.0, maxiter=200)
            update_best(final2)
        
        # Final SLSQP squeeze
        if time.time() < deadline - 2 and best_val > 0:
            slsqp_final = _slsqp_polish(final2, best_val * 0.999, maxiter=80)
            update_best(slsqp_final)
    
    # Convert best barycentric to Cartesian and return
    if best_bary is not None:
        return bary_to_cart(best_bary)
    else:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        return bary_to_cart(bary)
>>>>>>> REPLACE
