<<<<<<< SEARCH
        # 4. Second expansion pass after SLSQP
        if best_result is not None and time.time() < deadline - 10:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 5. Final high-precision polish with tighter tolerance
        if best_result is not None and time.time() < deadline - 4:
            # Use a very tight SLSQP polish
            def _tight_polish(z_):
                try:
                    res = minimize(
                        lambda zz: _vectorized_objective(zz, penalty_weight=50000.0),
                        z_,
                        method='SLSQP',
                        bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                        options={'maxiter': 500, 'ftol': 1e-16, 'maxls': 100}
                    )
                    return res.x
                except Exception:
                    return None
            
            polished2 = _tight_polish(best_result)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2
            
            # One more expansion pass after the tight polish
            if best_result is not None and time.time() < deadline - 1:
                expanded_final = _expansion_relaxation_polish(
                    best_result, 
                    time_budget=min(3.0, deadline - (time.time() - start_time) - 0.5)
                )
                if expanded_final is not None:
                    valid, val = check_validity(expanded_final)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded_final
=======
        # 4. Second expansion pass after SLSQP with bisection
        if best_result is not None and time.time() < deadline - 10:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 5. Final high-precision polish with tighter tolerance
        if best_result is not None and time.time() < deadline - 4:
            # Use a very tight SLSQP polish
            def _tight_polish(z_):
                try:
                    res = minimize(
                        lambda zz: _vectorized_objective(zz, penalty_weight=50000.0),
                        z_,
                        method='SLSQP',
                        bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                        options={'maxiter': 500, 'ftol': 1e-16, 'maxls': 100}
                    )
                    return res.x
                except Exception:
                    return None
            
            polished2 = _tight_polish(best_result)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2
            
            # One more expansion pass after the tight polish with high-precision bisection
            if best_result is not None and time.time() < deadline - 1:
                expanded_final = _expansion_relaxation_polish(
                    best_result, 
                    time_budget=min(4.0, deadline - (time.time() - start_time) - 0.5)
                )
                if expanded_final is not None:
                    valid, val = check_validity(expanded_final)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded_final

            # Final bisection squeeze: try to push each radius to its absolute limit
            if best_result is not None and time.time() < deadline - 0.5:
                z_final = best_result.copy()
                x_f = z_final[0::3][:N].copy()
                y_f = z_final[1::3][:N].copy()
                r_f = z_final[2::3][:N].copy()
                w_f = z_final[-1]
                h_f = 2.0 - w_f
                
                for i in range(N):
                    if time.time() > deadline - 0.3:
                        break
                    new_r = _bisection_max_radius(i, x_f, y_f, r_f, w_f, h_f, lo=r_f[i], tol=1e-11, max_iter=45)
                    if new_r > r_f[i] + 1e-10:
                        r_f[i] = new_r
                
                z_final[2::3][:N] = r_f
                valid, val = check_validity(z_final)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_final
>>>>>>> REPLACE

