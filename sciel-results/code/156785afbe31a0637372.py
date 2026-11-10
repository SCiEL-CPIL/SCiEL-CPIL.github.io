<<<<<<< SEARCH
def _global_optimize(centers, max_time):
    """Run L-BFGS-B on the repulsion potential."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # L-BFGS-B is efficient for this
    try:
        res = minimize(
            _repulsion_potential, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
    except Exception as e:
        # If minimize fails, return original
        return centers
        
    return res.x.reshape(N, 3)
=======
def _global_optimize(centers, max_time):
    """Run L-BFGS-B on the soft-min objective to maximize minimum distance."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use the smooth soft-min objective
    try:
        res = minimize(
            _soft_min_objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        if time.time() - start_time < max_time and res.x is not None:
            # Check if we have time for a second pass with higher beta
            r1 = _compute_radius(res.x.reshape(N, 3))
            remaining = max_time - (time.time() - start_time)
            if remaining > 5.0:
                # Second pass with higher beta for sharper min
                try:
                    res2 = minimize(
                        lambda x: _soft_min_objective(x, beta=100.0),
                        res.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
                    )
                    if res2.x is not None:
                        r2 = _compute_radius(res2.x.reshape(N, 3))
                        if r2 > r1:
                            return res2.x.reshape(N, 3)
                except Exception:
                    pass
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    # Fallback to repulsion potential if soft-min fails
    try:
        res = minimize(
            _repulsion_potential, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers
        
    return centers
>>>>>>> REPLACE

