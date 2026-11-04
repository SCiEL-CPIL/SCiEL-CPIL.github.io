<<<<<<< SEARCH
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations with a small gap to ensure validity
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Phase 1: L-BFGS-B with smooth objective to navigate non-convex landscape
            try:
                res_lbgfs = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 500
                    }
                )
                
                # Check if L-BFGS-B result is valid
                if is_valid(res_lbgfs.x):
                    L_val = get_L(res_lbgfs.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_lbgfs.x.copy()
                        
            except Exception:
                pass
            
            # Phase 2: Nelder-Mead polish on the best result so far
            if time.time() - start_time > DEADLINE - 15:
                continue
                
            try:
                # Use best_params if available, otherwise use L-BFGS-B result
                polish_start = best_params if best_params is not None else init_params
                
                res_nm = minimize(
                    objective,
                    polish_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                # Check if NM result is valid
                if is_valid(res_nm.x):
                    L_val = get_L(res_nm.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
                        
            except Exception:
                pass
    
    # Final high-precision polish pass
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        try:
            res_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
=======
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations with a small gap to ensure validity
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    # Add more diverse initial configs
    configs.append(make_hex_grid(3, 3, gap=gap))
    configs.append(make_hex_grid(5, 2, gap=gap))
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 10.0, 20.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Phase 1: L-BFGS-B with smooth objective to navigate non-convex landscape
            try:
                res_lbgfs = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 500
                    }
                )
                
                # Check if L-BFGS-B result is valid
                if is_valid(res_lbgfs.x):
                    L_val = get_L(res_lbgfs.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_lbgfs.x.copy()
                        
            except Exception:
                pass
            
            # Phase 2: Nelder-Mead polish on the best result so far
            if time.time() - start_time > DEADLINE - 20:
                continue
                
            try:
                # Use best_params if available, otherwise use L-BFGS-B result
                polish_start = best_params if best_params is not None else init_params
                
                res_nm = minimize(
                    objective,
                    polish_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                # Check if NM result is valid
                if is_valid(res_nm.x):
                    L_val = get_L(res_nm.x)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_nm.x.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Optimize outer rotation independently for the best configuration
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 15:
        try:
            res_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                L_val = get_L(res_final.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # Phase 5: One more outer rotation optimization after final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
>>>>>>> REPLACE
