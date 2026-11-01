<<<<<<< SEARCH
def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    """
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is fixed so that g(0) = 0.
    """
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
=======
def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _grid_search_coarse():
    """Dense grid search over a reasonable range to find good starting points."""
    best_val = np.inf
    best_params = None
    
    # Coarse grid: a0 in [-3, 1], a4 in [-0.2, 0.3], a8 in [-0.005, 0.005]
    # Use log-spaced or linear grid
    a0_vals = np.linspace(-3.0, 1.0, 25)
    a4_vals = np.linspace(-0.2, 0.3, 25)
    a8_vals = np.linspace(-0.005, 0.005, 15)
    
    for a0 in a0_vals:
        for a4 in a4_vals:
            for a8 in a8_vals:
                if time.time() > _DEADLINE - 60:
                    return best_val, best_params
                params = np.array([a0, a4, a8])
                val = _objective(params)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = params.copy()
    
    return best_val, best_params

def _grid_search_fine(center, radius, n_per_dim):
    """Fine grid search around a center point."""
    best_val = np.inf
    best_params = None
    
    a0_vals = np.linspace(center[0] - radius[0], center[0] + radius[0], n_per_dim)
    a4_vals = np.linspace(center[1] - radius[1], center[1] + radius[1], n_per_dim)
    a8_vals = np.linspace(center[2] - radius[2], center[2] + radius[2], n_per_dim)
    
    for a0 in a0_vals:
        for a4 in a4_vals:
            for a8 in a8_vals:
                if time.time() > _DEADLINE - 45:
                    return best_val, best_params
                params = np.array([a0, a4, a8])
                val = _objective(params)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = params.copy()
    
    return best_val, best_params

def construct():
    """
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 coefficient is fixed so that g(0) = 0.
    """
    best_val = np.inf
    best_params = None
    
    # Stage 1: Coarse grid search to locate global minimum region
    if time.time() < _DEADLINE - 80:
        gv, gp = _grid_search_coarse()
        if gp is not None:
            best_val = gv
            best_params = gp
    
    # Stage 2: Fine grid search around the best coarse point
    if best_params is not None and time.time() < _DEADLINE - 50:
        # Estimate radius from coarse grid spacing
        radius = np.array([0.1, 0.02, 0.001])
        fv, fp = _grid_search_fine(best_params, radius, 15)
        if fp is not None:
            best_val = fv
            best_params = fp
    
    # Stage 3: L-BFGS-B local optimization from multiple starts
    starts = [best_params] if best_params is not None else []
    # Add some heuristic starts
    heuristic_starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-2.0, 0.05, 0.0005]),
    ]
    for hs in heuristic_starts:
        if hs not in starts:
            starts.append(hs)
    
    # Also add random starts
    rng = np.random.default_rng(42)
    for i in range(10):
        starts.append(rng.normal(0, 0.5, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 4: Tight Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-18, 'fatol': 1e-20,
                                   'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 5: Final L-BFGS-B polish with very tight tolerances
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
>>>>>>> REPLACE
