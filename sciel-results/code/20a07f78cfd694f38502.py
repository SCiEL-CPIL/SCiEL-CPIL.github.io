<<<<<<< SEARCH
                S_found, params_found = _binary_search_min_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.0, max_iters=8
                )
                
                if params_found is not None and S_found < best_S:
                    # Final polish
                    polished = _polish(params_found, S_found, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
                
                # Early exit if we found a very good solution
                if best_S < 3.35:
                    break
=======
                S_found, params_found = _binary_search_min_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.0, max_iters=10
                )
                
                if params_found is not None and S_found < best_S:
                    # Final polish
                    polished = _polish(params_found, S_found, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
                
                # Early exit if we found a very good solution
                if best_S < 3.30:
                    break
>>>>>>> REPLACE

