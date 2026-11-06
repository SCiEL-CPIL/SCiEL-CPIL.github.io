<<<<<<< SEARCH
def _simulated_annealing(objective, x0, bounds, max_iter=5000, T0=1.0, T1=0.01, time_limit=None, start_time=None):
    """Simulated Annealing optimization."""
    n = len(x0)
    if time_limit is None:
        time_limit = 60.0
    if start_time is None:
        start_time = time.time()
    
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    x_best = x.copy()
    f_best = f_current
    
    T = T0
    cooling_factor = (T1 / T0) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a new candidate by perturbing one or more dimensions
        # Randomly select a few dimensions to perturb
        n_perturb = max(1, int(n * 0.1))
        indices = np.random.choice(n, n_perturb, replace=False)
        
        x_new = x.copy()
        for idx in indices:
            # Perturbation size proportional to range
            range_size = bounds[idx][1] - bounds[idx][0]
            step = range_size * 0.1 * np.random.randn()
            x_new[idx] += step
            # Clip to bounds
            x_new[idx] = np.clip(x_new[idx], bounds[idx][0], bounds[idx][1])
        
        f_new = objective(x_new)
        
        # Acceptance criterion
        if f_new < f_current:
            x = x_new
            f_current = f_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            # Accept with probability exp(-(f_new - f_current) / T)
            delta_f = f_new - f_current
            if delta_f < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_f / T)
            
            if np.random.rand() < accept_prob:
                x = x_new
                f_current = f_new
        
        T *= cooling_factor
    
    return x_best, f_best
=======
def _nelder_mead(objective, x0, bounds, time_limit=None, start_time=None):
    """Nelder-Mead optimization using SciPy."""
    from scipy.optimize import minimize
    
    if time_limit is None:
        time_limit = 60.0
    if start_time is None:
        start_time = time.time()
        
    n = len(x0)
    
    # Define a wrapper that handles bounds by penalizing or clipping
    # For NM, it's better to let it go slightly out of bounds and penalize, or clip.
    # Clipping is safer for physical constraints.
    def obj_wrapper(x):
        if time.time() - start_time > time_limit:
            return objective(x) # Return current value to stop gracefully if possible, or just compute
        # Clip to bounds
        x_clipped = np.clip(x, 0, 400) # max_layer_nm is 400
        # Check total thickness
        total = np.sum(x_clipped)
        if total > 1000: # max_total_nm
            # Scale down to fit
            scale = 1000.0 / total
            x_clipped = x_clipped * scale
        return objective(x_clipped)

    # Initial simplex
    # Use a small initial step size
    initial_step = 10.0
    
    try:
        res = minimize(
            obj_wrapper, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 2000,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
    except Exception:
        return x0, objective(x0)
        
    x_opt = res.x
    f_opt = objective(np.clip(x_opt, 0, 400))
    
    return x_opt, f_opt
>>>>>>> REPLACE

