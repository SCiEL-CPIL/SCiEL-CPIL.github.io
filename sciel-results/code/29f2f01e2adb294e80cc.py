def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_S = np.zeros(N+1, dtype=bool)
    best_size = 0
    
    for seed in range(50):
        if time.time() > deadline: break
        
        rng = np.random.RandomState(seed)
        constraints = {}
        S = np.ones(N+1, dtype=bool)
        
        # Greedy Init
        for p in _primes_list:
            if p > np.sum(S): break
            counts = np.array([np.sum(S[r::p]) for r in range(p)])
            if np.min(counts) > 0:
                min_c = np.min(counts)
                cands = np.where(counts == min_c)[0]
                r = int(cands[rng.randint(len(cands))])
                constraints[p] = r
                S[r::p] = False
        
        # Local Search
        current_constraints = constraints.copy()
        current_S = S.copy()
        current_size = np.sum(current_S)
        
        no_improve = 0
        while time.time() < deadline and no_improve < 100:
            # Try moves
            move_found = False
            # 1. Try removing constraints
            keys = list(current_constraints.keys())
            rng.shuffle(keys)
            for p in keys:
                # Apply removal
                r = current_constraints[p]
                # Restore S elements for this residue
                # We need to know which elements were removed ONLY by this constraint?
                # No, S is the intersection of complements.
                # If we remove constraint p, S_new = S_old U {x in S_old_removed_by_p AND not removed by others}
                # This is hard to compute incrementally without reconstructing.
                # Reconstructing is O(N log log N).
                # Let's just reconstruct.
                
                new_constraints = {k: v for k, v in current_constraints.items() if k != p}
                new_S = np.ones(N+1, dtype=bool)
                for pp, rr in new_constraints.items():
                    new_S[rr::pp] = False
                new_size = np.sum(new_S)
                if new_size > current_size:
                    if _is_admissible(new_S, _primes_list):
                        current_constraints = new_constraints
                        current_S = new_S
                        current_size = new_size
                        no_improve = 0
                        move_found = True
                        break
                # Revert logic not needed if we use new_S
            
            if move_found: continue
            
            # 2. Try changing constraints
            keys = list(current_constraints.keys())
            rng.shuffle(keys)
            for p in keys:
                old_r = current_constraints[p]
                # Try a random few new residues
                for _ in range(3):
                    new_r = rng.randint(p)
                    if new_r == old_r: continue
                    
                    new_constraints = current_constraints.copy()
                    new_constraints[p] = new_r
                    
                    new_S = np.ones(N+1, dtype=bool)
                    for pp, rr in new_constraints.items():
                        new_S[rr::pp] = False
                    new_size = np.sum(new_S)
                    
                    if new_size > current_size:
                        if _is_admissible(new_S, _primes_list):
                            current_constraints = new_constraints
                            current_S = new_S
                            current_size = new_size
                            no_improve = 0
                            move_found = True
                            break
                if move_found: break
            
            if not move_found:
                no_improve += 1
        
        if current_size > best_size:
            best_size = current_size
            best_S = current_S.copy()
            
    return [int(x) for x in np.where(best_S)[0]]
