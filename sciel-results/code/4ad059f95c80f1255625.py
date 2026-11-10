<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin (evaluator kills at 150s)
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Phase 1: Grid-based relaxation to quickly find a good region
    remaining = deadline - time.time()
    if remaining > 30.0:
        try:
            x_relaxed = _relaxation_pass(x0, n_iters=300, cell_size=0.25, dt=0.002)
            if np.all(np.isfinite(x_relaxed)) and np.all(np.linalg.norm(x_relaxed, axis=1) < 1.0):
                score = _calc_score(x_relaxed)
                if score > best_score:
                    best_score = score
                    best_x = x_relaxed.copy()
        except Exception:
            pass
    
    current_x = best_x.copy()
    
    # Phase 2: Progressive alpha schedule with soft-min optimization
    alphas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0]
    
    for alpha in alphas:
        remaining = deadline - time.time()
        if remaining < 15.0:  # Leave time for final polish
            break
            
        try:
            # Adjust maxiter based on remaining time
            max_iter = min(500, int(remaining * 2))
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
            )
            x_new = res.x.reshape(N, 2)
            
            # Validate: check all finite, inside container
            if not np.all(np.isfinite(x_new)):
                continue
            if np.any(np.linalg.norm(x_new, axis=1) >= 1.0):
                continue
                
            score = _calc_score(x_new)
            if score > best_score:
                best_score = score
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very high alpha
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(500.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
            )
            x_new = res.x.reshape(N, 2)
            if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 1.0):
                score = _calc_score(x_new)
                if score > best_score:
                    best_score = score
                    best_x = x_new.copy()
        except Exception:
            pass
    
    # Phase 4: One more relaxation + polish cycle if time permits
    remaining = deadline - time.time()
    if remaining > 10.0:
        try:
            x_relaxed2 = _relaxation_pass(best_x, n_iters=100, cell_size=0.2, dt=0.0005)
            if np.all(np.isfinite(x_relaxed2)) and np.all(np.linalg.norm(x_relaxed2, axis=1) < 1.0):
                score = _calc_score(x_relaxed2)
                if score > best_score:
                    best_score = score
                    best_x = x_relaxed2.copy()
                    current_x = x_relaxed2.copy()
        except Exception:
            pass
        
        remaining = deadline - time.time()
        if remaining > 5.0:
            try:
                res = minimize(
                    _soft_min_energy_and_grad,
                    current_x,
                    args=(800.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30}
                )
                x_new = res.x.reshape(N, 2)
                if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 1.0):
                    score = _calc_score(x_new)
                    if score > best_score:
                        best_score = score
                        best_x = x_new.copy()
            except Exception:
                pass
    
    # Return the best valid solution
    return best_x
=======
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(_RNG_SEED)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Base hex lattice
    initializations.append(_hex_lattice_init())
    
    # 2-5. Perturbed hex lattices with increasing perturbation
    for pert in [0.03, 0.05, 0.08, 0.12]:
        initializations.append(_perturbed_hex_init(rng, perturbation=pert))
    
    # 6-10. Random initializations
    for _ in range(5):
        initializations.append(_random_init(rng))
    
    best_x = initializations[0].copy()
    best_score = _calc_score(best_x)
    
    # Progressive alpha schedule
    alphas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0]
    
    # For each initialization, run progressive soft-min optimization
    for init_idx, x0 in enumerate(initializations):
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        # Estimate time per alpha stage
        time_per_alpha = max(2.0, remaining / (len(alphas) * 1.5))
        
        current_x = x0.copy()
        
        for alpha in alphas:
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            try:
                # Scale maxiter based on remaining time and alpha
                base_iter = 80
                if alpha <= 15.0:
                    max_iter = min(200, int(remaining / len(alphas) * 3))
                elif alpha <= 100.0:
                    max_iter = min(150, int(remaining / len(alphas) * 2.5))
                else:
                    max_iter = min(100, int(remaining / len(alphas) * 2))
                
                res = minimize(
                    _soft_min_energy_and_grad,
                    current_x,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 40
                    }
                )
                x_new = res.x.reshape(N, 2)
                
                # Validate
                if not np.all(np.isfinite(x_new)):
                    continue
                if np.any(np.linalg.norm(x_new, axis=1) >= 0.9999):
                    # Clamp and re-validate
                    radii = np.linalg.norm(x_new, axis=1)
                    mask = radii > 0.9999
                    if np.any(mask):
                        x_new[mask] *= 0.9999 / radii[mask]
                    if np.any(np.linalg.norm(x_new, axis=1) >= 0.9999):
                        continue
                
                score = _calc_score(x_new)
                if score > best_score:
                    best_score = score
                    best_x = x_new.copy()
                
                current_x = x_new.copy()
            except Exception:
                pass
    
    # Final high-precision polish on the best solution found
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                best_x,
                args=(1000.0,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 150,
                    'ftol': 1e-15,
                    'gtol': 1e-14,
                    'maxls': 30
                }
            )
            x_new = res.x.reshape(N, 2)
            if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 0.9999):
                score = _calc_score(x_new)
                if score > best_score:
                    best_score = score
                    best_x = x_new.copy()
        except Exception:
            pass
    
    # Return the best valid solution
    return best_x
>>>>>>> REPLACE
