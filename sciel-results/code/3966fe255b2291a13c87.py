<<<<<<< SEARCH
def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
        _get_perturbed_hex_lattice(N, seed=1),
        _get_perturbed_hex_lattice(N, seed=2),
    ]
    
    # Optimize each initialization with both methods
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Try L-BFGS-B optimization
        if time.time() - start_time < DEADLINE - 20:
            optimized_lbfgs = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized_lbfgs)):
                r = _compute_score(optimized_lbfgs)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_lbfgs.copy()
        
        # Try Nelder-Mead optimization
        if time.time() - start_time < DEADLINE - 15:
            optimized_nm = _optimize_nelder_mead(init, start_time)
            if np.all(np.isfinite(optimized_nm)):
                r = _compute_score(optimized_nm)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_nm.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=10000, initial_step=0.001)
        
    # High-beta refinement to sharpen the minimum
    if time.time() - start_time < DEADLINE - 5:
        refined, refined_score = _refine_with_high_beta(best_centers, start_time, max_betas=5)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined
            
    # Perturbation-retry loop to escape local minima
    if time.time() - start_time < DEADLINE - 10:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, start_time, n_perturbations=3)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=5000, initial_step=0.0005)
        
    return best_centers
# EVOLVE-BLOCK-END
=======
def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
        _get_perturbed_hex_lattice(N, seed=1),
        _get_perturbed_hex_lattice(N, seed=2),
    ]
    
    # First pass: try hybrid optimization on diverse initializations
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Try hybrid optimization (force-directed + L-BFGS-B + subgradient)
        if time.time() - start_time < DEADLINE - 20:
            hybrid_centers, hybrid_score = _hybrid_optimize(init, start_time)
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_centers = hybrid_centers.copy()
        
        # Also try pure L-BFGS-B for comparison
        if time.time() - start_time < DEADLINE - 15:
            optimized_lbfgs = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized_lbfgs)):
                r = _compute_score(optimized_lbfgs)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_lbfgs.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=10000, initial_step=0.001)
        
    # High-beta refinement to sharpen the minimum
    if time.time() - start_time < DEADLINE - 5:
        refined, refined_score = _refine_with_high_beta(best_centers, start_time, max_betas=5)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined
            
    # Perturbation-retry loop to escape local minima
    if time.time() - start_time < DEADLINE - 10:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, start_time, n_perturbations=3)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=5000, initial_step=0.0005)
        
    return best_centers
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
