<<<<<<< SEARCH
    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        # Fallback if greedy timed out or failed
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run randomized greedy multiple times to find a better starting point
    num_runs = 200 # Increased runs due to faster greedy
    for i in range(1, num_runs):
        if time.time() > deadline - 10.0:
            break
            
        sol = run_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
=======
    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        # Fallback if greedy timed out or failed
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run randomized greedy multiple times to find a better starting point
    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 30.0:
            break
            
        sol = run_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Maximal t-packing scaffold: select blocks with minimal overlap with existing solution
    # This provides a structural scaffold that covers disjoint t-subsets efficiently
    def run_packing_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Among max-gain candidates, prefer those with less overlap with existing solution
            # to maximize disjoint coverage
            if len(sol) > 0:
                # Compute overlap with existing solution blocks
                existing_ts = set()
                for b_idx in sol:
                    for ts_idx in block_to_ts[b_idx]:
                        existing_ts.add(ts_idx)
                
                scored_candidates = []
                for b_idx in candidates[:200]:  # Limit for speed
                    overlap = 0
                    for ts_idx in block_to_ts[b_idx]:
                        if ts_idx in existing_ts:
                            overlap += 1
                    score = -overlap  # Lower overlap is better
                    scored_candidates.append((score, b_idx))
                
                scored_candidates.sort(reverse=True)
                b_idx = scored_candidates[0][1]
            else:
                b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            # Update coverage
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
            
            # Update gains
            newly_covered = [ts_idx for ts_idx in block_to_ts[b_idx]]
            if newly_covered:
                blocks_to_decrement = []
                for ts_idx in newly_covered:
                    if not is_uncovered[ts_idx]:  # Was just covered
                        blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if blocks_to_decrement:
                    counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000
        
        # Convert to list of blocks
        return [list(all_k_blocks[b_idx]) for b_idx in sol]
    
    # Run packing greedy to get a better starting point
    for i in range(20):
        if time.time() > deadline - 25.0:
            break
        sol = run_packing_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
>>>>>>> REPLACE
