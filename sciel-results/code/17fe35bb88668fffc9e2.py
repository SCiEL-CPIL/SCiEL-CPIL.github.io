<<<<<<< SEARCH
    rng = np.random.RandomState(42)
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Random permutations
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, best_perm)
    
    # Run insertion local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 20:  # Leave 20s for SA
            break
        
        perm, ms = _insertion_local_search(pt, cand, deadline - 20, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run simulated annealing from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, best_perm, deadline - 5, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:3]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, cand, deadline - 2, max_iter=10000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    return best_perm
=======
    rng = np.random.RandomState(42)
    
    # Pre-extract job rows as Python lists for faster access in makespan computation
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Random permutations
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, pt_rows, best_perm)
    
    # Run insertion local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 20:  # Leave 20s for SA
            break
        
        perm, ms = _insertion_local_search(pt, pt_rows, cand, deadline - 20, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run simulated annealing from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, pt_rows, best_perm, deadline - 5, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:3]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, pt_rows, cand, deadline - 2, max_iter=10000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    return best_perm
>>>>>>> REPLACE

The NEH function also calls `_makespan`, so I need to update it:

