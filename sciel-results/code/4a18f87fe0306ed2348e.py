<<<<<<< SEARCH
def _optimize_smooth(h0, max_iter=200, temperature_start=5.0, temperature_end=0.1):
    """
    Optimize using smooth soft-max surrogate with annealing temperature.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Anneal temperature from high to low
    n_anneal = 5
    for stage in range(n_anneal):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        t_frac = stage / max(1, n_anneal - 1)
        temperature = temperature_start * (temperature_end / temperature_start) ** t_frac
        
        # Run optimization with current temperature
        for _ in range(max_iter // n_anneal):
            if time.time() - _start_time > _TIME_LIMIT - 10:
                break
            
            # Compute gradient
            grad = _soft_max_grad(h, temperature)
            
            # Gradient ascent with step size
            step_size = 0.01
            h_new = h + step_size * grad
            h_new = np.clip(h_new, 0, None)
            
            # Check improvement
            s_new = _soft_max_score(h_new, temperature)
            s_old = _soft_max_score(h, temperature)
            
            if s_new > s_old:
                h = h_new
            else:
                # Reduce step size
                step_size *= 0.5
            
            # Update best
            s_actual = _score(h)
            if s_actual > best_score:
                best_score = s_actual
                best_h = h.copy()
    
    return best_h, best_score
=======
def _optimize_smooth(h0, max_iter=200, temperature_start=5.0, temperature_end=0.1):
    """
    Optimize using smooth soft-max surrogate with annealing temperature.
    Uses line search for better convergence.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Anneal temperature from high to low
    n_anneal = 6
    for stage in range(n_anneal):
        if time.time() - _start_time > _TIME_LIMIT - 15:
            break
        
        t_frac = stage / max(1, n_anneal - 1)
        temperature = temperature_start * (temperature_end / temperature_start) ** t_frac
        
        # Run optimization with current temperature
        iters_per_stage = max(10, max_iter // n_anneal)
        for it in range(iters_per_stage):
            if time.time() - _start_time > _TIME_LIMIT - 15:
                break
            
            # Compute gradient
            grad = _soft_max_grad(h, temperature)
            
            if np.allclose(grad, 0):
                break
            
            # Gradient ascent with adaptive step size and line search
            step_size = 0.1 / (1.0 + it * 0.05)
            improved = False
            
            for ls_iter in range(15):
                h_new = h + step_size * grad
                h_new = np.clip(h_new, 0, None)
                
                s_new = _soft_max_score(h_new, temperature)
                s_old = _soft_max_score(h, temperature)
                
                if s_new > s_old * (1 + 1e-10):
                    h = h_new
                    improved = True
                    break
                else:
                    step_size *= 0.5
            
            if not improved:
                # Try smaller step
                step_size = 0.01
                for ls_iter in range(10):
                    h_new = h + step_size * grad
                    h_new = np.clip(h_new, 0, None)
                    s_new = _soft_max_score(h_new, temperature)
                    s_old = _soft_max_score(h, temperature)
                    if s_new > s_old * (1 + 1e-10):
                        h = h_new
                        break
                    step_size *= 0.5
            
            # Update best
            s_actual = _score(h)
            if s_actual > best_score:
                best_score = s_actual
                best_h = h.copy()
    
    return best_h, best_score
>>>>>>> REPLACE

