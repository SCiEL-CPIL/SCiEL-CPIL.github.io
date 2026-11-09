        # --- Phase 2: Nelder-Mead Polishing ---
        # Initialize Simplex
        delta = 0.1
        simplex = np.empty((d + 1, d))
        simplex[0] = best_x
        for i in range(d):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += delta
        
        f_simplex = np.array([eval_func(x) for x in simplex])
        
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main NM Loop
        while calls < budget:
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            if calls >= budget: break
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = eval_func(x_r)
            
            if f_r < f_simplex[0]:
                # Expansion
                if calls >= budget: break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = eval_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if calls >= budget: break
                x_oc = centroid + 0.5 * (simplex[-1] - centroid)
                x_oc = np.clip(x_oc, -5, 5)
                f_oc = eval_func(x_oc)
                if f_oc < f_simplex[-1]:
                    simplex[-1] = x_oc
                    f_simplex[-1] = f_oc
                else:
                    if calls >= budget: break
                    x_ic = centroid + 0.5 * (centroid - simplex[-1])
                    x_ic = np.clip(x_ic, -5, 5)
                    f_ic = eval_func(x_ic)
                    if f_ic < f_simplex[-1]:
                        simplex[-1] = x_ic
                        f_simplex[-1] = f_ic
                    else:
                        # Shrink
                        # Need d evaluations
                        if calls + d > budget:
                            break
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = eval_func(simplex[i])
            
            if np.max(np.abs(simplex - simplex[0])) < 1e-6:
                break
