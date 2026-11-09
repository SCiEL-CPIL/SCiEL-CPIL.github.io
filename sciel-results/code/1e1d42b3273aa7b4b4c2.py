<<<<<<< SEARCH
    time_remaining = deadline - time.time()
    
    if time_remaining > 10.0:
        # Run Hill Climbing
        # Use 40% of remaining time for HC, 40% for SA, 10% for polish, 10% buffer
        t_hc = time_remaining * 0.4
        if t_hc > 5.0:
            optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_hc, seed=42)
            sc_hc = max(max(r) for r in optimized_hc)
            if sc_hc < best_scope:
                best_scope = sc_hc
                best_rows = optimized_hc
        
        # Run SA on the current best (which might be improved by HC)
        t_sa = deadline - time.time() - 15.0 # Leave 15s for polish + buffer
        if t_sa > 5.0:
            optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc_sa = max(max(r) for r in optimized_sa)
            if sc_sa < best_scope:
                best_scope = sc_sa
                best_rows = optimized_sa
        
        # Final polish pass
        t_polish = deadline - time.time() - 5.0 # Leave 5s buffer
        if t_polish > 1.0:
            polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
            sc_polish = max(max(r) for r in polished)
            if sc_polish < best_scope:
                best_scope = sc_polish
                best_rows = polished
                
    return best_rows
=======
    time_remaining = deadline - time.time()
    
    if time_remaining > 10.0:
        # Run Hill Climbing with multiple seeds for diversity
        # Use 35% of remaining time for HC, 40% for SA, 15% for polish, 10% buffer
        t_hc = time_remaining * 0.35
        if t_hc > 5.0:
            # Try HC with two different seeds
            for hc_seed in [42, 43]:
                if time.time() > deadline - (time_remaining * 0.35 / 2 + 15):
                    break
                t_this_hc = min(t_hc / 2, deadline - time.time() - 15)
                if t_this_hc > 3.0:
                    optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_this_hc, seed=hc_seed)
                    sc_hc = max(max(r) for r in optimized_hc)
                    if sc_hc < best_scope:
                        best_scope = sc_hc
                        best_rows = optimized_hc
        
        # Run SA on the current best (which might be improved by HC)
        t_sa = deadline - time.time() - 20.0 # Leave 20s for polish + buffer
        if t_sa > 5.0:
            # Try SA with two different seeds
            for sa_seed in [44, 45]:
                if time.time() > deadline - (t_sa / 2 + 20):
                    break
                t_this_sa = min(t_sa / 2, deadline - time.time() - 20)
                if t_this_sa > 3.0:
                    optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_this_sa, seed=sa_seed)
                    sc_sa = max(max(r) for r in optimized_sa)
                    if sc_sa < best_scope:
                        best_scope = sc_sa
                        best_rows = optimized_sa
        
        # Final polish pass
        t_polish = deadline - time.time() - 5.0 # Leave 5s buffer
        if t_polish > 1.0:
            polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
            sc_polish = max(max(r) for r in polished)
            if sc_polish < best_scope:
                best_scope = sc_polish
                best_rows = polished
                
    return best_rows
>>>>>>> REPLACE
