def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using three stages:
    1. L-BFGS-B on the smooth soft-min surrogate.
    2. SLSQP on the epigraph formulation (maximize t s.t. areas >= t).
    3. Nelder-Mead on the hard minimum.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        return bary_init
        
    if time.time() - start_time > time_limit * 0.4:
        return clamp_bary(result1.x.reshape(N, 2))
        
    bary_mid = clamp_bary(result1.x.reshape(N, 2))
    
    # Stage 2: SLSQP on Epigraph
    try:
        min_area = -hard_min_obj(bary_mid.flatten())
        x0 = np.concatenate([bary_mid.flatten(), [min_area]])
        
        def slsqp_obj(x):
            return -x[-1]
            
        def slsqp_area_cons(x):
            b = x[:22].reshape(N, 2)
            areas = calc_areas_bary(b)
            return areas - x[-1]
            
        def slsqp_sum_cons(x):
            u = x[0::2]
            v = x[1::2]
            return 1.0 - (u + v)
            
        constraints = [
            {'type': 'ineq', 'fun': slsqp_area_cons},
            {'type': 'ineq', 'fun': slsqp_sum_cons}
        ]
        
        slsqp_bounds = bounds + [(0, 1.0)]
        
        result_slsqp = minimize(
            slsqp_obj,
            x0,
            method='SLSQP',
            bounds=slsqp_bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if result_slsqp.success:
             bary_slsqp = clamp_bary(result_slsqp.x[:22].reshape(N, 2))
             val_slsqp = -hard_min_obj(bary_slsqp.flatten())
             val_mid = -hard_min_obj(bary_mid.flatten())
             if val_slsqp > val_mid:
                 bary_mid = bary_slsqp
                 
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_mid
        
    # Stage 3: Nelder-Mead on hard min
    try:
        result2 = minimize(
            hard_min_obj,
            bary_mid.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16}
        )
    except Exception:
        return bary_mid
        
    return clamp_bary(result2.x.reshape(N, 2))
