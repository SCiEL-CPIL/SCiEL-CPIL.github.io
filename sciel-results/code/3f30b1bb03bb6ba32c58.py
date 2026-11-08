def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Known good starting point
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_C4 = _objective_and_grad(best_c[0], best_c[1], best_c[2])[0]
    
    # Differential Evolution for global search
    bounds = [(-100, 100), (-100, 100), (-100, 100)]
    
    def obj_func(x):
        val, _ = _objective_and_grad(x[0], x[1], x[2])
        return val
    
    try:
        de_result = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=400, 
            popsize=20, 
            tol=1e-12, 
            seed=42,
            init='sobol'
        )
        
        if de_result.fun < best_C4:
            best_C4 = de_result.fun
            best_c = de_result.x
    except Exception:
        pass
    
    # Local polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
            
    return np.array(best_c)
