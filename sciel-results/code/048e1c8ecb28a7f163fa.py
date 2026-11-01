def construct():
    """Construct the packing configuration."""
    
    # Initial guesses
    # 1. Linear spacing
    x0_1 = _initial_guess()
    
    # 2. Square root spacing (area based)
    cum_counts = np.cumsum(ring_counts)
    radii_sq = np.sqrt(cum_counts) / np.sqrt(N) * 0.95
    phases_zero = np.zeros(num_rings)
    x0_2 = np.concatenate([radii_sq, phases_zero])
    
    # 3. Randomized
    np.random.seed(42)
    radii_rand = np.random.uniform(0.1, 0.95, num_rings)
    radii_rand = np.sort(radii_rand)
    phases_rand = np.random.uniform(0, 2 * np.pi, num_rings)
    x0_3 = np.concatenate([radii_rand, phases_rand])
    
    # 4. Hexagonal-like spacing
    # Approximate radius for ring i is proportional to i
    radii_hex = np.linspace(0.05, 0.95, num_rings)
    phases_hex = np.zeros(num_rings)
    x0_4 = np.concatenate([radii_hex, phases_hex])
    
    # 5. Perturbed Hexagonal
    np.random.seed(123)
    radii_hex_pert = radii_hex + np.random.uniform(-0.05, 0.05, num_rings)
    radii_hex_pert = np.clip(radii_hex_pert, 0.01, 0.99)
    phases_hex_pert = np.random.uniform(0, 2 * np.pi, num_rings)
    x0_5 = np.concatenate([radii_hex_pert, phases_hex_pert])

    starts = [x0_1, x0_2, x0_3, x0_4, x0_5]
    
    best_points = None
    best_radius = -1.0
    
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    # Time limit management
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    for i, x0 in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        # Use Nelder-Mead
        # Reduce maxiter for later starts if time is tight, but 2000 is usually fine
        max_iter = 2000 if i == 0 else 1500
        
        try:
            result = minimize(
                _objective, x0, args=(ring_counts,),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            best_params = result.x
            radii = np.maximum(best_params[:num_rings], 0.0)
            phases = best_params[num_rings:]
            
            points = _build_points(radii, phases, ring_counts)
            
            if len(points) != N:
                continue
                
            r = _compute_radius(points)
            
            if r > best_radius:
                best_radius = r
                best_points = points
                
        except Exception as e:
            pass
            
    if best_points is None:
        # Fallback to the original single run logic or a simple heuristic
        x0 = _initial_guess()
        result = minimize(
            _objective, x0, args=(ring_counts,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_params = result.x
        radii = np.maximum(best_params[:num_rings], 0.0)
        phases = best_params[num_rings:]
        best_points = _build_points(radii, phases, ring_counts)
        
    # Final check and return
    if len(best_points) != N:
         # Emergency fallback
         spacing = 2.0 / (np.sqrt(N) + 1.0)
         while True:
             k = int(1.0 / spacing) + 2
             rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                     for i in range(-k, k + 1) for j in range(-k, k + 1)]
             pts = np.array(rows)
             radius = np.hypot(pts[:, 0], pts[:, 1])
             pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
             if len(pts) >= N:
                 return pts[np.argsort(radius, kind="stable")[:N]]
             spacing *= 0.98
             
    return best_points
