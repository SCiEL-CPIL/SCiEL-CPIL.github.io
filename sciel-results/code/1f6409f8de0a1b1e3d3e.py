<<<<<<< SEARCH
    # Final L-BFGS-B polish with very tight soft-min
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        x0 = best_centers.flatten()
        
        def obj(x):
            return _soft_min_obj(x, 0.005)
        
        def obj_grad(x):
            return _soft_min_grad(x, 0.005)
        
        try:
            result = opt.minimize(
                obj, x0, jac=obj_grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.fun < obj(x0):
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                polished_r = _compute_radius_fast(polished)
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
        except Exception:
            pass
        
        # Second polish with even tighter tau
        remaining = _DEADLINE - time.time()
        if remaining > 5:
            x0 = best_centers.flatten()
            
            def obj2(x):
                return _soft_min_obj(x, 0.002)
            
            def obj2_grad(x):
                return _soft_min_grad(x, 0.002)
            
            try:
                result2 = opt.minimize(
                    obj2, x0, jac=obj2_grad, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 10}
                )
                if result2.fun < obj2(x0):
                    polished2 = result2.x.reshape(-1, 2)
                    polished2 = np.clip(polished2, -0.5, 0.5)
                    polished2_r = _compute_radius_fast(polished2)
                    if polished2_r > best_radius:
                        best_centers = polished2
                        best_radius = polished2_r
            except Exception:
                pass
=======
    # Final L-BFGS-B polish with very tight soft-min
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        x0 = best_centers.flatten()
        
        # Use a slightly larger tau first to escape local minima, then tighten
        tau_values = [0.01, 0.005, 0.002, 0.001]
        
        for tau in tau_values:
            if time.time() > _DEADLINE - 5:
                break
            
            def obj(x):
                return _soft_min_obj(x, tau)
            
            def obj_grad(x):
                return _soft_min_grad(x, tau)
            
            try:
                # Increase maxiter since gradient is faster
                result = opt.minimize(
                    obj, x0, jac=obj_grad, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
                )
                if result.fun < obj(x0):
                    polished = result.x.reshape(-1, 2)
                    polished = np.clip(polished, -0.5, 0.5)
                    polished_r = _compute_radius_fast(polished)
                    if polished_r > best_radius:
                        best_centers = polished
                        best_radius = polished_r
                        x0 = polished.flatten() # Use improved result for next tau
            except Exception:
                pass
>>>>>>> REPLACE

