def _tabu_search(perm, pt, time_limit=10.0):
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    current_perm = perm.copy()
    current_ms = _makespan_vec(current_perm, pt)
    global_best_perm = current_perm.copy()
    global_best_ms = current_ms
    
    tabu_size = max(5, n // 2)
    tabu_list = []
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        best_move_ms = float('inf')
        best_move_i = -1
        best_move_j = -1
        best_trial = None
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            job = current_perm[i]
            base = list(current_perm)
            del base[i]
            
            for j in range(n):
                if i == j:
                    continue
                
                # Tabu check
                is_tabu = False
                for t in tabu_list:
                    if t[0] == i and t[1] == j:
                        is_tabu = True
                        break
                
                # Construct trial
                insert_idx = j if j < i else j - 1
                trial = base[:insert_idx] + [job] + base[insert_idx:]
                
                ms = _makespan_vec(np.array(trial, dtype=int), pt)
                
                if is_tabu and ms >= global_best_ms:
                    continue
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_trial = trial
        
        if best_move_i == -1:
            break
            
        tabu_list.append((best_move_i, best_move_j))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        
        new_perm = np.array(best_trial, dtype=int)
        
        if best_move_ms < global_best_ms:
            global_best_ms = best_move_ms
            global_best_perm = new_perm.copy()
            
        current_perm = new_perm
        current_ms = best_move_ms
        
    return global_best_perm
