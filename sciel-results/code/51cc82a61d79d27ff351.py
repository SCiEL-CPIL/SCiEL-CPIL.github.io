<<<<<<< SEARCH
    p0 = np.concatenate([X0, Y0])

    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    best_p = p0.copy()
    best_v = objective(p0)

    # Use Differential Evolution for global search
    # Bounds: X and Y should be in [-1, 1] roughly
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        res_de = so.differential_evolution(objective, bounds, maxiter=200, popsize=25, seed=42, polish=True, tol=1e-8)
        if time.time() < deadline:
            if res_de.fun < best_v:
                best_v = res_de.fun
                best_p = res_de.x.copy()
    except Exception:
        pass

    # Polish with Powell
    if time.time() < deadline - 90:
        try:
            res_pw = so.minimize(objective, best_p, method='Powell', options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12})
            if res_pw.fun < best_v:
                best_v = res_pw.fun
                best_p = res_pw.x.copy()
        except Exception:
            pass

    # Additional local refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            res_nm = so.minimize(objective, best_p, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < best_v:
                best_v = res_nm.fun
                best_p = res_nm.x.copy()
        except Exception:
            pass

    # Final high-precision polish with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            res_lbfgs = so.minimize(objective, best_p, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if res_lbfgs.fun < best_v:
                best_v = res_lbfgs.fun
                best_p = res_lbfgs.x.copy()
        except Exception:
            pass
=======
    p0 = np.concatenate([X0, Y0])

    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin (evaluator kills at 630s)

    best_p = p0.copy()
    best_v = objective(p0)

    def polish(p, time_limit):
        """Multi-stage local polish with time budget."""
        t0 = time.time()
        # Powell (robust, derivative-free)
        if time.time() < time_limit - 20:
            try:
                r = so.minimize(objective, p, method='Powell',
                                options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-14})
                if r.fun < best_v:
                    best_v = r.fun
                    p = r.x.copy()
            except Exception:
                pass
        # Nelder-Mead (handles non-smoothness)
        if time.time() < time_limit - 10:
            try:
                r = so.minimize(objective, p, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12,
                                         'adaptive': True})
                if r.fun < best_v:
                    best_v = r.fun
                    p = r.x.copy()
            except Exception:
                pass
        # L-BFGS-B final precision
        if time.time() < time_limit - 2:
            try:
                r = so.minimize(objective, p, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18})
                if r.fun < best_v:
                    best_v = r.fun
                    p = r.x.copy()
            except Exception:
                pass
        return p, best_v

    # Phase 1: Differential Evolution (global)
    bounds = [(-1.5, 1.5)] * (2 * N)
    de_time_limit = deadline - 200  # reserve 200s for multi-restart + polish
    if time.time() < de_time_limit:
        try:
            res_de = so.differential_evolution(
                objective, bounds, maxiter=150, popsize=20, seed=42,
                polish=True, tol=1e-10, mutation=(0.5, 1.0), recombination=0.7)
            if res_de.fun < best_v:
                best_v = res_de.fun
                best_p = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Multi-restart local optimization from diverse starting points
    rng = np.random.default_rng(7)
    n_restarts = 8
    for ri in range(n_restarts):
        if time.time() > deadline - 120:
            break
        # Generate diverse starting points
        if ri == 0:
            start = best_p.copy()
        elif ri == 1:
            start = p0.copy()  # original Keich
        elif ri == 2:
            # Symmetric: X[i] = -Y[i]
            sym = np.zeros(2 * N)
            sym[:N] = rng.uniform(-0.5, 0.5, N)
            sym[N:] = -sym[:N]
            start = sym
        elif ri == 3:
            # Antisymmetric with offset
            off = rng.uniform(-0.3, 0.3, N)
            start = np.concatenate([off, -off + 0.1 * rng.uniform(-1, 1, N)])
        else:
            start = rng.uniform(-1.0, 1.0, 2 * N)

        # Quick polish with tighter budget per restart
        remaining = deadline - 120 - (time.time() - start_time)
        per_restart_budget = min(15.0, remaining / max(1, n_restarts - ri))
        if per_restart_budget > 3.0:
            p_r, v_r = polish(start, start_time + time.time() - start_time + per_restart_budget)
            if v_r < best_v:
                best_v = v_r
                best_p = p_r.copy()

    # Phase 3: Final high-precision polish on the best solution
    if time.time() < deadline - 5:
        best_p, best_v = polish(best_p, deadline)

    # Phase 4: Ultra-tight final polish
    if time.time() < deadline - 2:
        try:
            r = so.minimize(objective, best_p, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14,
                                     'adaptive': True})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
        try:
            r = so.minimize(objective, best_p, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-20})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
>>>>>>> REPLACE
