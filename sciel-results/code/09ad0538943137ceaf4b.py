<<<<<<< SEARCH
    # Phase 2: Basin Hopping
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=50,
                T=1.0,
                stepsize=0.5,
                minimizer_kwargs={'options': {'maxiter': 100}},
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x
        except Exception:
            pass
=======
    # Phase 2: Basin Hopping
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=80,
                T=1.5,
                stepsize=0.8,
                minimizer_kwargs={'options': {'maxiter': 200}},
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x
        except Exception:
            pass
>>>>>>> REPLACE

