<<<<<<< SEARCH
    # More beta stages with finer granularity
    betas = [15.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0]
    
    for beta in betas:
        if time.time() > deadline - 3: # Leave more time for final checks
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        try:
            # Use tighter tolerances for higher betas
            maxiter = 400 if beta >= 100 else 300
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 60}
            )
        except Exception:
            continue

        if time.time() > deadline - 3:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])
        
        # Clamp to ensure validity before checking
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        if _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj - 1e-10:
            # Even if not strictly better, keep it if very close (may help subsequent stages)
            current_pos = pos_opt.copy()
=======
    # More beta stages with finer granularity and aggressive start
    betas = [10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline - 2: # Leave time for final checks
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        try:
            # Use tighter tolerances for higher betas
            maxiter = 600 if beta >= 100 else 400
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 80}
            )
        except Exception:
            continue

        if time.time() > deadline - 2:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])
        
        # Clamp to ensure validity before checking
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        if _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj - 1e-10:
            # Even if not strictly better, keep it if very close (may help subsequent stages)
            current_pos = pos_opt.copy()
>>>>>>> REPLACE

