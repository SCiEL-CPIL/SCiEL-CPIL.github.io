<<<<<<< SEARCH
                # 2. Try full optimization if time permits
                if time.time() < DEADLINE - 30:
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = init_pts[i, 0]
                        init_params[3*i+1] = init_pts[i, 1]
                        init_params[3*i+2] = 0.0
                    
                    S_found, params_found = _binary_search_min_S(
                        init_params, ctr, oa, DEADLINE,
                        S_lo=3.0, S_hi=5.5, max_iters=7
                    )
                    
                    if params_found is not None and S_found < best_S:
                        # Final polish
                        polished = _refine_local(params_found, S_found, ctr, oa, DEADLINE)
                        if _check_validity(polished, S_found, ctr, oa):
                            best_S = S_found
                            best_result = (S_found, polished, ctr, oa)
                        else:
                            best_S = S_found
                            best_result = (S_found, params_found, ctr, oa)
=======
                # 2. Try full optimization if time permits
                if time.time() < DEADLINE - 30:
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = init_pts[i, 0]
                        init_params[3*i+1] = init_pts[i, 1]
                        init_params[3*i+2] = 0.0
                    
                    S_found, params_found = _binary_search_min_S(
                        init_params, ctr, oa, DEADLINE,
                        S_lo=3.0, S_hi=5.5, max_iters=12
                    )
                    
                    if params_found is not None and S_found < best_S:
                        # Final polish with higher penalty power
                        polished = _refine_local(params_found, S_found, ctr, oa, DEADLINE, penalty_power=3.0)
                        if _check_validity(polished, S_found, ctr, oa):
                            best_S = S_found
                            best_result = (S_found, polished, ctr, oa)
                        else:
                            best_S = S_found
                            best_result = (S_found, params_found, ctr, oa)
>>>>>>> REPLACE

