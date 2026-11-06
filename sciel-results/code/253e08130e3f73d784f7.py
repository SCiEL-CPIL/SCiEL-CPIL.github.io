<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
    ]
    
    # Z samples for evaluation
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = _gauss_legendre_nodes(n)
    
    # Convert to log-parameterization
    # Inverse of tanh: atanh(2*x - 1)
    log_nodes_init = np.arctanh(2.0 * gl_nodes - 1.0)
    # Inverse of softmax: just use log of weights (up to constant)
    log_weights_init = np.log(gl_weights)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Evaluate initial score
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # L-BFGS-B optimization
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective, params_init,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Differential evolution with multiple restarts
    remaining = deadline - time.time()
    if remaining > 30:
        bounds = [(-5, 5)] * n + [(-5, 5)] * n
        
        def de_objective(params):
            return _objective(params, z_samples, models)
        
        # Multiple restarts
        n_restarts = 3
        time_per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=50, popsize=15,
                    tol=1e-6, seed=42 + r,
                    polish=True,
                    workers=1,
                    time_limit=time_per_restart - 2
                )
                
                if result.fun < best_score:
                    best_params = result.x
                    best_nodes = _log_param_to_nodes(result.x[:n])
                    best_weights = _log_param_to_weights(result.x[n:])
                    best_score = result.fun
            except Exception as e:
                pass
    
    # Final refinement with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective, best_params,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
        (0.05, 0.1, -0.95),     # Additional diversity
        (0.15, -0.2, -1.05),    # Additional diversity
    ]
    
    # Z samples for evaluation - denser coverage
    z_samples = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = _gauss_legendre_nodes(n)
    
    # Convert to log-parameterization
    # Inverse of tanh: atanh(2*x - 1)
    log_nodes_init = np.arctanh(2.0 * gl_nodes - 1.0)
    # Inverse of softmax: just use log of weights (up to constant)
    log_weights_init = np.log(gl_weights)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Evaluate initial score
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # Phase 1: Variance-based node pre-optimization
    # Find nodes that maximize information content (variance of 1/E across models)
    if time.time() < deadline - 40:
        try:
            # Use differential evolution to find high-variance nodes
            bounds_nodes = [(-6, 6)] * n
            
            result_var = differential_evolution(
                _variance_objective, bounds_nodes,
                maxiter=30, popsize=12,
                tol=1e-4, seed=123,
                polish=False,
                workers=1,
                time_limit=15
            )
            
            # Use these as initial nodes for main optimization
            if result_var.fun < 0:  # Negative means we found variance
                var_nodes = _log_param_to_nodes(result_var.x)
                # Keep GL weights initially
                params_init = np.concatenate([result_var.x, log_weights_init])
                best_nodes = var_nodes.copy()
                best_weights = gl_weights.copy()
                
                # Re-evaluate
                errors = []
                for z in z_samples:
                    for wa, wb, w0 in models:
                        err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
                        errors.append(err)
                best_score = np.max(errors)
        except Exception:
            pass
    
    # Phase 2: Smooth L-BFGS-B optimization using log-sum-exp surrogate
    if time.time() < deadline - 25:
        try:
            result = minimize(
                _smooth_objective, params_init,
                args=(z_samples, models, 50.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 3: Differential evolution with multiple restarts
    remaining = deadline - time.time()
    if remaining > 25:
        bounds = [(-6, 6)] * n + [(-4, 4)] * n
        
        def de_objective(params):
            return _objective(params, z_samples, models)
        
        # Multiple restarts with varied seeds
        n_restarts = min(4, max(1, int(remaining // 10)))
        time_per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=40, popsize=15,
                    tol=1e-6, seed=42 + r * 7,
                    polish=True,
                    workers=1,
                    time_limit=max(5, time_per_restart - 3)
                )
                
                if result.fun < best_score:
                    best_params = result.x
                    best_nodes = _log_param_to_nodes(result.x[:n])
                    best_weights = _log_param_to_weights(result.x[n:])
                    best_score = result.fun
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish with least_squares
    if time.time() < deadline - 3:
        try:
            # Use least_squares for a smooth local optimization
            result_ls = least_squares(
                _residuals_for_ls, best_params,
                args=(z_samples, models),
                method='trf',
                ftol=1e-14, xtol=1e-14, gtol=1e-14,
                max_nfev=2000
            )
            
            # Check if it improved
            final_errs = []
            for z in z_samples:
                for wa, wb, w0 in models:
                    err = _relative_error(z, _log_param_to_nodes(result_ls.x[:n]), 
                                          _log_param_to_weights(result_ls.x[n:]), wa, wb, w0)
                    final_errs.append(err)
            ls_score = np.max(final_errs)
            
            if ls_score < best_score:
                best_params = result_ls.x
                best_nodes = _log_param_to_nodes(result_ls.x[:n])
                best_weights = _log_param_to_weights(result_ls.x[n:])
                best_score = ls_score
        except Exception:
            pass
    
    # Phase 5: Ultra-precise final L-BFGS-B polish
    if time.time() < deadline - 1:
        try:
            result = minimize(
                _smooth_objective, best_params,
                args=(z_samples, models, 200.0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    # Enforce strict increasing with minimum spacing
    for i in range(1, len(best_nodes)):
        if best_nodes[i] <= best_nodes[i-1] + 1e-12:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    # Clip to valid range
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
>>>>>>> REPLACE
