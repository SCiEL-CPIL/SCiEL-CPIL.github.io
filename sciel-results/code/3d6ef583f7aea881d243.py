<<<<<<< SEARCH
    # Local Search: Try to replace 2 blocks with 1, or 1 with 1?
    # Since we want to minimize count, we look for:
    # 1. Redundant blocks (already done in current code, but we restart it).
    # 2. Swap: Replace block A with block B if B covers all t-subsets that A uniquely covered?
    #    No, B must cover the union of unique parts of A and any other block?
    #    Simpler: If we remove A, which t-subsets become uncovered? 
    #    If there is a block B (not in set) that covers ALL of those, we can swap A->B?
    #    No, B might not be in the set. We can ADD B and REMOVE A?
    #    That keeps count same.
    #    To reduce count, we need to find a block B that covers the "unique" t-subsets of A, 
    #    AND B is already in the set? No.
    #    We need to find a set of blocks (size < size of removed) that covers the gap.
    #    Gap = t-subsets covered ONLY by A.
    #    If Gap is empty, A is redundant.
    #    If Gap is not empty, can we cover Gap with 0 blocks? No.
    #    Can we cover Gap with 1 block B, and remove A?
    #    If yes, count stays same.
    #    To reduce count, we need to remove A and NOT add any block.
    #    This implies A's unique coverage was already covered by others. (Redundancy).
    
    # So, to reduce count, we just need to find redundancies.
    # The current code does this.
    # Why is the current score only 1.1189?
    # Maybe the greedy is not optimal and B&B could find a better one.
    
    # Let's try a limited B&B with a strict time limit.
    
    start_time = time.time()
    
    # We will use a heuristic B&B that picks the hardest t-subset.
    # To make it fast, we pre-compute the list of blocks for each t-subset?
    # Memory: 11628 lists of avg length 75582 * (8/19) ~ 3000.
    # 11628 * 3000 ints = 35 million ints. ~280 MB. Might be too much/slow to build.
    
    # Alternative:
    # Just return the greedy result with redundancy removal.
    # The current program DOES this.
    # So I must do something else.
    
    # Let's try a "Greedy with Randomization" or "Multi-start Greedy".
    
    best_found = selected_greedy[:]
    
    for _ in range(10): # 10 restarts
        uncovered = all_mask
        selected = []
        used = [False] * len(k_subsets)
        
        while uncovered:
            # Randomized Greedy: Pick from top 5 best gains
            gains = []
            for i in range(len(k_subsets)):
                if not used[i]:
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
            
            if not gains:
                break
            
            gains.sort(reverse=True)
            
            # Pick from top min(5, len(gains))
            top_k = gains[:min(5, len(gains))]
            # Weighted random or just pick best?
            # Let's pick best to ensure progress, but maybe shuffle ties?
            # Or pick random from top 10%?
            
            # To improve, let's pick the best one (standard greedy) but 
            # maybe we can try a different tie-breaking or slight perturbation.
            
            # Actually, let's just use the standard greedy again?
            # No, let's try to pick a random block from the top 20 candidates with gain > 0.
            
            candidates = [i for g, i in gains if g >= gains[0][0] - 1] # Close to best
            if not candidates:
                candidates = [i for g, i in gains[:5]]
                
            idx = candidates[0] # Default to best
            
            selected.append(idx)
            used[idx] = True
            uncovered &= ~masks[idx]
            
        # Remove redundants
        selected = try_remove_redundant(selected)
        
        if len(selected) < len(best_found):
            best_found = selected[:]
            
        if time.time() - start_time > 60:
            break
            
    return [list(k_subsets[i]) for i in best_found]
=======
    def optimize_solution(sol):
        # sol is a list of block indices
        n_blocks = len(sol)
        if n_blocks == 0:
            return sol
            
        # cover_count[j] = number of blocks in sol that cover t-subset j
        cover_count = [0] * n_t_subsets
        
        # Initialize cover counts
        for b_idx in sol:
            mask = masks[b_idx]
            # Iterate through set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        # Try to remove blocks that are not critical
        current_sol = sol[:]
        changed = True
        while changed:
            changed = False
            for i in range(len(current_sol) - 1, -1, -1):
                b_idx = current_sol[i]
                mask = masks[b_idx]
                
                # Check if any t-subset in this block has cover_count == 1
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if not is_critical:
                    # Remove block
                    current_sol.pop(i)
                    # Update cover counts
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
        
        return current_sol

    best_found = optimize_solution(selected_greedy)
    
    start_time = time.time()
    time_limit = 110.0
    
    for _ in range(20): # 20 restarts
        if time.time() - start_time > time_limit:
            break
            
        uncovered = all_mask
        selected = []
        used = [False] * len(k_subsets)
        
        while uncovered:
            gains = []
            for i in range(len(k_subsets)):
                if not used[i]:
                    gain = (masks[i] & uncovered).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
            
            if not gains:
                break
            
            gains.sort(reverse=True)
            
            # Randomized selection from top candidates
            top_n = min(10, len(gains))
            # Use random choice to break symmetry
            idx = random.choice([g[1] for g in gains[:top_n]])
            
            selected.append(idx)
            used[idx] = True
            uncovered &= ~masks[idx]
            
        optimized = optimize_solution(selected)
        
        if len(optimized) < len(best_found):
            best_found = optimized[:]
            
    return [list(k_subsets[i]) for i in best_found]
>>>>>>> REPLACE
