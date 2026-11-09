<<<<<<< SEARCH
def _optimize_for_radius(pos_init, r_target):
    """
    Minimize the violation for a fixed radius r_target.
    Returns the optimized positions and the minimum violation.
    """
    def objective(x):
        pos = x.reshape(N, 2)
        return _violation(pos, r_target)
        
    x0 = pos_init.flatten()
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
    
    pos_opt = res.x.reshape(N, 2)
    viol = res.fun
    
    return pos_opt, viol
=======
def _optimize_for_radius(pos_init, r_target):
    """
    Minimize the violation for a fixed radius r_target.
    Returns the optimized positions and the minimum violation.
    """
    def objective(x):
        pos = x.reshape(N, 2)
        return _violation(pos, r_target)
        
    x0 = pos_init.flatten()
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
    
    pos_opt = res.x.reshape(N, 2)
    viol = res.fun
    
    return pos_opt, viol

def _hybrid_optimize(pos_init, r_target, time_budget=10.0):
    """
    Hybrid optimization: MD simulation followed by local polish.
    """
    start_time = time.time()
    
    # Phase 1: MD simulation to escape local minima
    pos_md = _md_simulate(pos_init, r_target, max_steps=20000, dt_initial=1e-4,
                          k_repulsive=100.0, k_boundary=50.0, damping=0.999, temperature=0.01)
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.7:
        # Not enough time for polish, just use MD result
        r_actual = _compute_actual_radius(pos_md)
        return pos_md, _violation(pos_md, r_target)
    
    # Phase 2: Local polish with high precision
    pos_polished, viol = _local_polish(pos_md, r_target, max_iter=500, ftol=1e-14, gtol=1e-12)
    
    return pos_polished, viol
>>>>>>> REPLACE

