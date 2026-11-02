<<<<<<< SEARCH
def _try_rank_hybrid(r, n, m, p, T_target, deadline, seed=0):
    """Try both L-BFGS and Adam, return best result with polish pass."""
    results = []
    
    # Try L-BFGS first
    if time.time() < deadline - 40:
        res = _try_rank_lbfgs(r, n, m, p, T_target, deadline, seed=seed, maxiter=2000)
        if res is not None:
            results.append(res)
    
    # Try Adam with multiple seeds
    for s in range(4):
        if time.time() > deadline - 25:
            break
        res = _try_rank_adam(r, n, m, p, T_target, deadline, seed=seed + s * 7919, niter=7000)
        if res is not None:
            results.append(res)
    
    if not results:
        return None
    
    # Find best candidate
    best = None
    best_err = float('inf')
    for U, V, W in results:
        T = _reconstruct(U, V, W)
        err = np.max(np.abs(T - T_target))
        if err < best_err:
            best_err = err
            best = (U, V, W)
    
    # Polish pass: if close, do a final high-precision optimization
    if best is not None and best_err < 1e-3 and time.time() < deadline - 20:
        U, V, W = best
        # Use L-BFGS for final polish with tight tolerances
        params = _flatten(U, V, W)
        try:
            from scipy.optimize import minimize
            result = minimize(
                _objective,
                params,
                args=(n, m, p, r, T_target),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            params_polished = result.x
            U_p, V_p, W_p = _unflatten(params_polished, n, m, p, r)
            T_p = _reconstruct(U_p, V_p, W_p)
            err_p = np.max(np.abs(T_p - T_target))
            if err_p < best_err:
                best = (U_p, V_p, W_p)
                best_err = err_p
        except Exception:
            pass
    
    if best is not None and best_err < 1e-5:
        return best
    return None
=======
def _try_rank_hybrid(r, n, m, p, T_target, deadline, seed=0):
    """Try both L-BFGS and Adam, return best result with polish pass."""
    results = []
    
    # Try L-BFGS first
    if time.time() < deadline - 45:
        res = _try_rank_lbfgs(r, n, m, p, T_target, deadline, seed=seed, maxiter=3000)
        if res is not None:
            results.append(res)
    
    # Try Adam with multiple seeds
    for s in range(5):
        if time.time() > deadline - 30:
            break
        res = _try_rank_adam(r, n, m, p, T_target, deadline, seed=seed + s * 7919, niter=8000)
        if res is not None:
            results.append(res)
    
    if not results:
        return None
    
    # Find best candidate
    best = None
    best_err = float('inf')
    for U, V, W in results:
        T = _reconstruct(U, V, W)
        err = np.max(np.abs(T - T_target))
        if err < best_err:
            best_err = err
            best = (U, V, W)
    
    # Polish pass: if close, do a final high-precision optimization
    if best is not None and best_err < 1e-3 and time.time() < deadline - 15:
        U, V, W = best
        # Use L-BFGS for final polish with tight tolerances
        params = _flatten(U, V, W)
        try:
            from scipy.optimize import minimize
            result = minimize(
                _objective,
                params,
                args=(n, m, p, r, T_target),
                method='L-BFGS-B',
                options={'maxiter': 8000, 'maxfun': 15000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            params_polished = result.x
            U_p, V_p, W_p = _unflatten(params_polished, n, m, p, r)
            T_p = _reconstruct(U_p, V_p, W_p)
            err_p = np.max(np.abs(T_p - T_target))
            if err_p < best_err:
                best = (U_p, V_p, W_p)
                best_err = err_p
        except Exception:
            pass
    
    if best is not None and best_err < 1e-5:
        return best
    return None
>>>>>>> REPLACE
