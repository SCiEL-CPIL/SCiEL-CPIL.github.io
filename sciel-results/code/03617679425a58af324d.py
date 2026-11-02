def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _voronoi_initial(s),
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
        lambda s: _relaxed_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            params = init_fn(seed)
            
            # Optimization with increasing penalty
            for pen in [100.0, 500.0, 2000.0, 5000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    res = minimize(_penalty_objective, params, jac=_penalty_grad, 
                                   args=(pen,), method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 200, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass
            
            # Enforce validity
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)
            
            if score > best_score:
                best_score = score
                best_params = valid_p.copy()
                
            # Polish step: Try to push radii out further from the valid state
            if time.time() < deadline - 5 and best_params is not None:
                try:
                    # Start from the best valid config found so far (or current if better)
                    start_p = best_params.flatten()
                    res = minimize(_penalty_objective, start_p, jac=_penalty_grad, 
                                   args=(10000.0,), method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 100, 'maxfun': 150, 'ftol': 1e-14, 'gtol': 1e-10})
                    polished_p = res.x
                    valid_polished = _enforce_validity(polished_p)
                    polished_score = _score_valid(valid_polished)
                    
                    if polished_score > best_score:
                        best_score = polished_score
                        best_params = valid_polished.copy()
                except Exception:
                    pass

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
