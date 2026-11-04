<<<<<<< SEARCH
        # Allocate time for this LS run
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
=======
        # Allocate time for this LS run, but don't exceed the ILS window
        il_deadline = time_budget - polish_time - 2
        duration = il_deadline - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Final polish pass: run local search on the best solution found
    polish_deadline = time_budget - 3  # 3s safety margin
    polish_duration = polish_deadline - (time.time() - start_time)
    if polish_duration > 2:
        p_polish, cost_polish = _local_search(a, b, best_p, polish_duration, start_time)
        if cost_polish < best_cost:
            best_cost = cost_polish
            best_p = p_polish.copy()
    
    return best_p.tolist()
>>>>>>> REPLACE
