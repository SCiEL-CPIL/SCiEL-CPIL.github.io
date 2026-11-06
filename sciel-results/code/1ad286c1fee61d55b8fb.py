<<<<<<< SEARCH
    # Run SA from top inits
    # We have ~30s left maybe? 
    # Let's allocate time.
    
    if time.time() < deadline - 5.0:
        best_init_score, best_init_A, best_init_counts = scored_inits[0]
        
        A_res, score_res = run_sa(best_init_A, best_init_counts, 50000, 50.0, 0.1, 0.999)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
            
    if time.time() < deadline - 5.0 and len(scored_inits) > 1:
        second_init_score, second_init_A, second_init_counts = scored_inits[1]
        if second_init_A != best_A:
             A_res, score_res = run_sa(second_init_A, second_init_counts, 30000, 50.0, 0.1, 0.999)
             if score_res < best_score:
                 best_score = score_res
                 best_A = A_res
=======
    # Run SA from top inits with load-balancing focus
    # Allocate time based on remaining budget
    
    remaining_time = deadline - time.time()
    num_sa_runs = min(3, max(1, int(remaining_time / 15)))
    
    for i in range(num_sa_runs):
        if time.time() < deadline - 8.0 and i < len(scored_inits):
            init_score, init_A, init_counts = scored_inits[i]
            
            # Vary SA parameters for diversity
            T_start = 30.0 + i * 10
            T_end = 0.05
            alpha = 0.9995 - i * 0.0001
            max_steps = int((deadline - 8.0 - time.time()) * 2000)
            max_steps = max(10000, min(max_steps, 80000))
            
            A_res, score_res = run_sa(init_A, init_counts, max_steps, T_start, T_end, alpha)
            
            if score_res < best_score:
                best_score = score_res
                best_A = A_res
>>>>>>> REPLACE

