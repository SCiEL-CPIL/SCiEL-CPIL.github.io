<<<<<<< SEARCH
def _optimize_from(w, r, xs, ys, beta_start=20.0, beta_end=200.0, n_steps=8, time_limit=None, deadline=None):
    """
    Run L-BFGS-B optimization with annealed beta.
    Returns best (w, r, xs, ys) found.
    """
    best_obj = np.inf
    best_v = None
    
    v = np.concatenate([[w, r], xs, ys])
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (step / max(n_steps - 1, 1))
        
        def obj_func(v):
            return _objective_and_grad(v, beta)[0]
        
        def grad_func(v):
            return _objective_and_grad(v, beta)[1]
        
        bounds = [(0.1, 1.9), (0.01, 0.5)] + [(-0.5, 2.5)] * (2*N)
        
        try:
            res = minimize(obj_func, v, jac=grad_func, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x.copy()
            
            v = res.x.copy()
        except Exception:
            break
    
    if best_v is not None:
        w_best = best_v[0]
        r_best = best_v[1]
        xs_best = best_v[2:2+N]
        ys_best = best_v[2+N:2+2*N]
        
        if _strictly_valid(w_best, 2.0 - w_best, r_best, xs_best, ys_best):
            return w_best, r_best, xs_best, ys_best
    
    return w, r, xs, ys
=======
def _optimize_from(w, r, xs, ys, beta_start=20.0, beta_end=200.0, n_steps=8, deadline=None, fixed_w=None):
    """
    Run L-BFGS-B optimization with annealed beta.
    Returns best (w, r, xs, ys) found.
    """
    best_obj = np.inf
    best_v = None
    
    v = np.concatenate([[w, r], xs, ys])
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (step / max(n_steps - 1, 1))
        
        def obj_func(v, beta=beta, fixed_w=fixed_w):
            return _objective_and_grad(v, beta, fixed_w)[0]
        
        def grad_func(v, beta=beta, fixed_w=fixed_w):
            return _objective_and_grad(v, beta, fixed_w)[1]
        
        if fixed_w is not None:
            bounds = [(fixed_w, fixed_w), (0.01, 0.5)] + [(-0.5, 2.5)] * (2*N)
        else:
            bounds = [(0.1, 1.9), (0.01, 0.5)] + [(-0.5, 2.5)] * (2*N)
        
        try:
            res = minimize(obj_func, v, jac=grad_func, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x.copy()
            
            v = res.x.copy()
        except Exception:
            break
    
    if best_v is not None:
        w_best = best_v[0]
        r_best = best_v[1]
        xs_best = best_v[2:2+N]
        ys_best = best_v[2+N:2+2*N]
        
        if _strictly_valid(w_best, 2.0 - w_best, r_best, xs_best, ys_best):
            return w_best, r_best, xs_best, ys_best
    
    return w, r, xs, ys
>>>>>>> REPLACE

