<<<<<<< SEARCH
def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_fibonacci_initialization(N),
    ]
    
    # First pass: optimize each initialization with moderate beta
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 35:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        optimized = _optimize_from_start(init, beta=50.0, maxiter=100, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 25:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: try to improve with perturbations of the best
    if best_centers is not None and time.time() - start_time < _DEADLINE - 15:
        rng = np.random.default_rng(7)
        for attempt in range(8):
            if time.time() - start_time > _DEADLINE - 15:
                break
            
            noise_scale = 0.005 + 0.01 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                noisy[mask] *= 0.98 / norms[mask]
            
            optimized = _optimize_from_start(noisy, beta=200.0, maxiter=60, gtol=1e-8)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > best_score:
                    best_score = r
                    best_centers = optimized.copy()
        
        # Final polish
        if time.time() - start_time < _DEADLINE - 5:
            best_centers, best_score = _polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
=======
def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_fibonacci_initialization(N),
    ]
    
    # First pass: optimize each initialization with moderate beta
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 40:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        optimized = _optimize_from_start(init, beta=50.0, maxiter=100, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: use rigid-body dynamics to find jammed states from multiple starts
    if best_centers is not None and time.time() - start_time < _DEADLINE - 30:
        # Try dynamics from best and a few other initializations
        dyn_starts = [best_centers]
        for init in initializations[:3]:
            norms = np.linalg.norm(init, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                init = init.copy()
                init[mask] *= 0.98 / norms[mask]
            dyn_starts.append(init)
        
        # Also try with slightly larger target radius to encourage expansion
        for idx, start_pos in enumerate(dyn_starts):
            if time.time() - start_time > _DEADLINE - 25:
                break
            
            # Use current best radius as target, with slight inflation
            r_target = best_score * 1.001
            dyn_result = _rigid_body_dynamics(start_pos, r_target, max_steps=3000, dt=0.002, seed=42+idx)
            
            if np.all(np.isfinite(dyn_result)):
                # Ensure inside container
                norms = np.linalg.norm(dyn_result, axis=1)
                max_allowed = 1.0 - r_target
                mask = norms > max_allowed
                if np.any(mask):
                    dyn_result = dyn_result.copy()
                    dyn_result[mask] *= max_allowed / norms[mask]
                
                r = _compute_score(dyn_result)
                if r > best_score:
                    best_score = r
                    best_centers = dyn_result.copy()
    
    # Third pass: polish the best solution with gradient methods
    if best_centers is not None and time.time() - start_time < _DEADLINE - 20:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Fourth pass: try to improve with perturbations + dynamics + polish
    if best_centers is not None and time.time() - start_time < _DEADLINE - 15:
        rng = np.random.default_rng(7)
        for attempt in range(6):
            if time.time() - start_time > _DEADLINE - 15:
                break
            
            noise_scale = 0.005 + 0.01 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                noisy[mask] *= 0.98 / norms[mask]
            
            # Try both gradient and dynamics
            optimized = _optimize_from_start(noisy, beta=200.0, maxiter=60, gtol=1e-8)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > best_score:
                    best_score = r
                    best_centers = optimized.copy()
            
            # Also try dynamics from perturbed start
            if time.time() - start_time < _DEADLINE - 18:
                r_target = best_score * 1.001
                dyn_result = _rigid_body_dynamics(noisy, r_target, max_steps=2000, dt=0.002, seed=100+attempt)
                if np.all(np.isfinite(dyn_result)):
                    norms = np.linalg.norm(dyn_result, axis=1)
                    max_allowed = 1.0 - r_target
                    mask = norms > max_allowed
                    if np.any(mask):
                        dyn_result = dyn_result.copy()
                        dyn_result[mask] *= max_allowed / norms[mask]
                    r = _compute_score(dyn_result)
                    if r > best_score:
                        best_score = r
                        best_centers = dyn_result.copy()
        
        # Final polish
        if time.time() - start_time < _DEADLINE - 5:
            best_centers, best_score = _polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
>>>>>>> REPLACE
