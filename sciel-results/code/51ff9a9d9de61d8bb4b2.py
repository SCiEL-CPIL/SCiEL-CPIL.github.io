def _local_search(perm, P, time_limit=5.0):
    n_jobs = len(perm)
    current_perm = list(perm)
    current_mk = _makespan(current_perm, P)
    start_time = time.time()
    no_improve = 0
    
    while no_improve < 500:
        if time.time() - start_time > time_limit:
            break
            
        move_type = random.randint(0, 1)
        improved = False
        
        if move_type == 0:
            # Or-opt
            seg_len = random.randint(1, min(3, n_jobs))
            i = random.randint(0, n_jobs - seg_len)
            segment = current_perm[i:i + seg_len]
            temp_perm = current_perm[:i] + current_perm[i + seg_len:]
            n_rem = len(temp_perm)
            
            best_pos = i
            best_mk = current_mk
            for j in range(n_rem + 1):
                if j == i: continue
                new_perm = temp_perm[:j] + segment + temp_perm[j:]
                mk = _makespan(new_perm, P)
                if mk < best_mk:
                    best_mk = mk
                    best_pos = j
            
            if best_mk < current_mk:
                current_perm = temp_perm[:best_pos] + segment + temp_perm[best_pos:]
                current_mk = best_mk
                improved = True
        else:
            # 2-opt
            i = random.randint(0, n_jobs - 1)
            j = random.randint(i + 1, n_jobs - 1)
            if j - i < 1: continue
            new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
            mk = _makespan(new_perm, P)
            if mk < current_mk:
                current_perm = new_perm
                current_mk = mk
                improved = True
                
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            
    return current_perm, current_mk
