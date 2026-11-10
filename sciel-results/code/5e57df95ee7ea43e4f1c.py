# Inside hexagon_packing_11

    best_params = initial_params
    best_score = objective(best_params)
    
    # Run optimization for a few restarts
    for restart in range(5):
        if time.time() - start_time > timeout - 60:
            break
            
        if restart == 0:
            current_params = initial_params
        else:
            # Perturb the best solution so far
            current_params = best_params + np.random.normal(0, 0.1, size=34)
            
        try:
            res = minimize(objective, current_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            pass
