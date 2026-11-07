<<<<<<< SEARCH
        # Phase 2: (1+1)-ES polish from best point found
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            # Main polish: adapt sigma aggressively
            polish_calls = 0
            max_polish = min(polish_budget, budget - calls_used)
            for _ in range(max_polish):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    update_best(x, fx)
                else:
                    sigma *= 0.9

            # Fine refinement: smaller steps for precision
            sigma = max(sigma, 0.05)
            fine_calls = 0
            max_fine = min(30, budget - calls_used)
            for _ in range(max_fine):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                fine_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    update_best(x, fx)
                else:
                    sigma *= 0.92
=======
        # Phase 2: Nelder-Mead polish from best point found
        if best_x is not None:
            nm_calls = 0
            max_nm = min(polish_budget, budget - calls_used)

            # Initialize simplex around best point
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                simplex[i + 1] = best_x.copy()
                simplex[i + 1, i] += 0.5

            # Evaluate simplex vertices
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                simplex_f[i] = func(simplex[i])
                calls_used += 1
                nm_calls += 1
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_nm_iter = max_nm - nm_calls

            for _ in range(max_nm_iter):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                nm_calls += 1

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    nm_calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    nm_calls += 1

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_used += 1
                            nm_calls += 1

                # Update best
                idx_min = np.argmin(simplex_f)
                if simplex_f[idx_min] < best_f:
                    best_f = simplex_f[idx_min]
                    best_x = simplex[idx_min].copy()

                # Check convergence
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < 1e-10:
                    break
>>>>>>> REPLACE

