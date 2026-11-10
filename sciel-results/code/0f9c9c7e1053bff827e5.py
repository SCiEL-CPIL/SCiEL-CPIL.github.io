<<<<<<< SEARCH
    # Optimize positions with multiple methods and restarts
    if time.time() - start_time < deadline - safety_margin:
        x0 = best_inner[:, :2].flatten()
        
        methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
        
        for method in methods:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            max_trials = 8
            for trial in range(max_trials):
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                if trial == 0:
                    x_init = x0.copy()
                else:
                    np.random.seed(trial * 1234 + 5678)
                    perturbation = 0.05 if trial < 4 else 0.02
                    x_init = x0 + np.random.randn(22) * perturbation
                
                try:
                    if method == 'L-BFGS-B':
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                    elif method == 'Nelder-Mead':
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-10, 'adaptive': True}
                        )
                    else:  # Powell
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
                        )
                    
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    
                    test_inner = np.zeros((11, 3))
                    for i in range(11):
                        test_inner[i, 0] = result.x[2 * i]
                        test_inner[i, 1] = result.x[2 * i + 1]
                        test_inner[i, 2] = 0.0
                    
                    if is_valid_configuration(test_inner, best_outer_angle):
                        new_side = calculate_min_side(test_inner, best_outer_angle)
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = test_inner
                            x0 = test_inner[:, :2].flatten()
                            
                except Exception:
                    pass
=======
    # Optimize positions with multiple methods and restarts
    if time.time() - start_time < deadline - safety_margin and best_inner is not None:
        x0 = best_inner[:, :2].flatten()
        
        methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
        
        for method in methods:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            max_trials = 8
            for trial in range(max_trials):
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                if trial == 0:
                    x_init = x0.copy()
                else:
                    np.random.seed(trial * 1234 + 5678)
                    perturbation = 0.05 if trial < 4 else 0.02
                    x_init = x0 + np.random.randn(22) * perturbation
                
                try:
                    if method == 'L-BFGS-B':
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                    elif method == 'Nelder-Mead':
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-10, 'adaptive': True}
                        )
                    else:  # Powell
                        result = minimize(
                            objective,
                            x_init,
                            args=(best_outer_angle,),
                            method=method,
                            options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
                        )
                    
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    
                    # Ensure result.x is 1D and length 22
                    res_x = np.asarray(result.x).flatten()
                    if res_x.shape[0] != 22:
                        continue

                    test_inner = np.zeros((11, 3))
                    for i in range(11):
                        test_inner[i, 0] = res_x[2 * i]
                        test_inner[i, 1] = res_x[2 * i + 1]
                        test_inner[i, 2] = 0.0
                    
                    if is_valid_configuration(test_inner, best_outer_angle):
                        new_side = calculate_min_side(test_inner, best_outer_angle)
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = test_inner
                            x0 = test_inner[:, :2].flatten()
                            
                except Exception:
                    pass
>>>>>>> REPLACE

I also need to ensure the final refinement handles `best_inner` being `None` (though it shouldn't be).

