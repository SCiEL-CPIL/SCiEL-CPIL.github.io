    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Try 3 restarts per aspect ratio
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Generate initial packing
            x0, y0, r0 = generate_hex_packing(w, h, N)
            
            # Add jitter for restart > 0
            if restart > 0:
                jitter = 0.05 * min(w, h)
                x0 += np.random.normal(0, jitter, N)
                y0 += np.random.normal(0, jitter, N)
                x0 = np.clip(x0, 0, w)
                y0 = np.clip(y0, 0, h)
                # Recalculate radii after jitter
                for i in range(N):
                    min_d = np.inf
                    min_d = min(min_d, x0[i], w-x0[i], y0[i], h-y0[i])
                    for j in range(N):
                        if i!=j:
                            d = np.sqrt((x0[i]-x0[j])**2 + (y0[i]-y0[j])**2)
                            min_d = min(min_d, d)
                    r0[i] = 0.4 * min_d

            # Check heuristic score
            if validate(x0, y0, r0, w, h):
                score_heur = np.sum(r0)
                if score_heur > best_score:
                    best_score = score_heur
                    best_result = np.column_stack([x0, y0, r0])
            
            # Prepare initial vector for optimizer
            x_init = np.zeros(3*N)
            for i in range(N):
                x_init[3*i] = x0[i]
                x_init[3*i+1] = y0[i]
                x_init[3*i+2] = r0[i]
                
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0, min(w, h)/2))
                
            try:
                # L-BFGS-B
                res = minimize(
                    objective, 
                    x_init, 
                    args=(w, h),
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )
                
                x_opt = res.x[0::3]
                y_opt = res.x[1::3]
                r_opt = res.x[2::3]
                
                # Clip radii
                r_opt = np.clip(r_opt, 0, None)
                
                if validate(x_opt, y_opt, r_opt, w, h):
                    score_opt = np.sum(r_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_result = np.column_stack([x_opt, y_opt, r_opt])
                        
                        # Polish
                        if time.time() < deadline - 2:
                            res2 = minimize(
                                objective, 
                                res.x, 
                                args=(w, h),
                                method='L-BFGS-B', 
                                bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-10}
                            )
                            x_pol = res2.x[0::3]
                            y_pol = res2.x[1::3]
                            r_pol = res2.x[2::3]
                            r_pol = np.clip(r_pol, 0, None)
                            
                            if validate(x_pol, y_pol, r_pol, w, h):
                                score_pol = np.sum(r_pol)
                                if score_pol > best_score:
                                    best_score = score_pol
                                    best_result = np.column_stack([x_pol, y_pol, r_pol])
                                    
            except Exception:
                pass
