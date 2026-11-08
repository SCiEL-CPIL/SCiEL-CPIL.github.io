    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        found = False
        for seed in range(10): # 10 attempts with different seeds
            if time.time() - start_time > DEADLINE - 5:
                break
                
            res = _try_optimize_rank(n, m, p, rank, seed=42 + seed)
            if res:
                best = res
                best_rank = rank
                found = True
                break # Move to next rank
                
        if rank == lower_bound and found:
            break
