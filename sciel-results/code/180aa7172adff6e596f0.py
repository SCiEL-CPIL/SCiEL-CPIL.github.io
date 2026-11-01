<<<<<<< SEARCH
        # Initial guess
        best_beta = 0.0
        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        
        if best_S_local == float('inf'):
            continue

        # Minimize over beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-8})
                if res.fun < best_S_local:
                    best_beta = res.x
                    best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                    
                    # Local polish
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
                        res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                              options={'maxiter': 500, 'xatol': 1e-10})
                        if res2.fun < best_S_local:
                            best_beta = res2.x
                            best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass
=======
        # Initial guess
        best_beta = 0.0
        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        
        if best_S_local == float('inf'):
            continue

        # Minimize over beta using smooth surrogate first, then polish with exact LP
        try:
            if _time_ok():
                # Phase 1: Coarse search with smooth objective
                res = minimize_scalar(smooth_obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-6})
                if res.fun < best_S_local * 1.01:  # Allow some tolerance
                    candidate_beta = res.x
                    # Verify with exact LP
                    cx_test, cy_test, S_test = solve_for_beta(candidate_beta)
                    if S_test < best_S_local:
                        best_beta = candidate_beta
                        best_cx, best_cy, best_S_local = cx_test, cy_test, S_test
                
                # Phase 2: Local polish with exact LP objective
                if _time_ok():
                    local_bounds = (max(0, best_beta - 0.05), min(math.pi / 6, best_beta + 0.05))
                    res2 = minimize_scalar(lambda b: solve_for_beta(b)[2], bounds=local_bounds, method='bounded',
                                          options={'maxiter': 500, 'xatol': 1e-10})
                    if res2.fun < best_S_local:
                        best_beta = res2.x
                        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                
                # Phase 3: Ultra-fine polish
                if _time_ok():
                    ultra_bounds = (max(0, best_beta - 0.001), min(math.pi / 6, best_beta + 0.001))
                    res3 = minimize_scalar(lambda b: solve_for_beta(b)[2], bounds=ultra_bounds, method='bounded',
                                          options={'maxiter': 1000, 'xatol': 1e-12})
                    if res3.fun < best_S_local:
                        best_beta = res3.x
                        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass
>>>>>>> REPLACE
