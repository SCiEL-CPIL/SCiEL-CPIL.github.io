<<<<<<< SEARCH
    # Use Nelder-Mead or L-BFGS-B
    # Since the objective is non-smooth (max of smooth functions), use Nelder-Mead
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0 + np.random.randn(39) * 0.5
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best result
    inner_pos = best_params[:24].reshape(12, 2)
    inner_angles = best_params[24:36]
    outer_center = best_params[36:38]
    outer_rotation = best_params[38]
    
    # Convert angles to degrees
    inner = []
    for i in range(12):
        inner.append([float(inner_pos[i, 0]), float(inner_pos[i, 1]), float(np.degrees(inner_angles[i]))])
    
    side_length = float(best_val)
    
    return inner, (float(outer_center[0]), float(outer_center[1])), side_length, float(np.degrees(outer_rotation))
=======
    # Use Nelder-Mead or L-BFGS-B
    # Since the objective is non-smooth (max of smooth functions), use Nelder-Mead
    best_params = x0.copy()
    best_side_len, best_penalty = eval_params(x0)
    best_val = best_side_len + 100.0 * best_penalty
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0 + np.random.randn(39) * 0.5
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best result
    inner_pos = best_params[:24].reshape(12, 2)
    inner_angles = best_params[24:36]
    outer_center = best_params[36:38]
    outer_rotation = best_params[38]
    
    # Re-evaluate to get the true side length (penalty should be ~0)
    final_side_len, final_penalty = eval_params(best_params)
    
    # Convert angles to degrees
    inner = []
    for i in range(12):
        inner.append([float(inner_pos[i, 0]), float(inner_pos[i, 1]), float(np.degrees(inner_angles[i]))])
    
    # Use the side length derived from the geometry, not the penalized objective
    side_length = float(final_side_len)
    
    return inner, (float(outer_center[0]), float(outer_center[1])), side_length, float(np.degrees(outer_rotation))
>>>>>>> REPLACE
