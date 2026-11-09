import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based greedy construction and local search.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Build conflict masks
    # conflict_mask[i] is an integer where bit j is set if i conflicts with j
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            # Check if candidate conflicts with any selected
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search(subset, mask):
        """
        Local search: try to swap elements to increase size.
        Returns improved subset and mask.
        """
        subset = list(subset)
        mask = mask
        improved = True
        while improved:
            improved = False
            n = len(subset)
            # Shuffle subset order to break symmetry
            random.shuffle(subset)
            
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                # Remove u
                trial_mask = mask & ~(1 << u)
                trial_subset = subset[:i] + subset[i+1:]
                
                # Find candidates to add
                # A candidate v is valid if (conflict_mask[v] & trial_mask) == 0 and v not in trial_subset
                # We iterate through all v not in trial_subset
                # Optimization: Precompute list of not-in-set? 
                # For m=205, iterating all 205 is fine.
                
                best_candidate = -1
                # We want to add a candidate. Any candidate works for size increase (1 removed, 1 added -> same size? No, we want to INCREASE size).
                # Wait. The current local search removes 1 and adds 1. Size stays same.
                # To increase size, we need to find a v that allows adding 1 without removing 1? 
                # No, greedy is maximal. To increase size, we must remove some and add more.
                # Standard 1-exchange keeps size same but might allow further exchanges.
                # To actually increase size, we need to find a "2-for-1" or similar.
                # However, if we just do 1-exchanges, we might get stuck in a local optimum of size K.
                # A better strategy: 
                # 1. Try to add any v not in set. If possible, add it. (This will never work if set is maximal).
                # 2. So we must remove elements.
                # Let's try: Remove 1, Add 1. This explores the space of maximal independent sets of the same size.
                # To increase size, we need a heuristic that finds a larger set.
                # Random restarts are the main driver for finding larger sizes.
                # Let's stick to robust random restarts + 1-exchange to polish.
                
                # Actually, if we just do random greedy restarts, we might find larger sets.
                # The 1-exchange helps to find a "better" local optimum (one that might allow easier expansion? No, size is fixed).
                # So the primary search is the number of restarts.
                
                # Let's try a different move: 
                # Pick a random subset of k elements to remove (k=1 or 2), then greedily add back as many as possible.
                # If k=1: Remove u, add best v. Size same.
                # If k=2: Remove u, v. Add w, x, y? 
                
                # For now, let's just do random restarts. It's simple and effective for small graphs.
                # We can increase the number of restarts.
                
                # Let's try to add a "kick": 
                # Remove a random element u.
                # Then try to add ANY element v not in set that is compatible.
                # If found, we have size N. (Same size).
                # This doesn't increase size.
                
                # To increase size, we need to find a set of size N+1.
                # This requires removing at least 1 element and adding at least 2?
                # Or removing 2 and adding 3?
                
                # Given the time, let's rely on many random greedy restarts.
                # The current program does 500. We can do 5000 if fast enough.
                # Greedy is O(N^2) or O(N * M) with bitmasks? 
                # Greedy with bitmasks: For each of 205 candidates, check 1 AND op. Very fast.
                # 205 * 205 operations per trial. ~40k ops.
                # 5000 trials = 200 million ops. Python might be slow.
                # Let's estimate: 1 trial ~ 1ms? 5000 trials ~ 5s.
                # So we can do many more.
                
                # Let's break the local search loop and just do restarts in the main loop.
                # The local search as written doesn't increase size.
                # So we should just return the greedy result.
                
                return subset, mask # No improvement logic for size increase in this simple version
                
        return subset, mask

    best_subset = []
    best_size = 0
    best_mask = 0
    
    random.seed(42)
    
    # Run many random greedy trials
    num_trials = 20000
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        subset, mask = greedy_construct(order)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            best_mask = mask
            
            # If we find a very large set, we might stop early? 
            # But we don't know the max.
            # Let's just keep going.
            
            # Optional: Run a local search that tries to increase size?
            # Since 1-exchange doesn't increase size, we need a different move.
            # Let's try "Remove 2, Add 3" heuristic?
            # Or just rely on restarts.
            
            # If best_size is high, maybe we can reduce trials?
            # No, let's use all time.
            
    # Final check
    if not best_subset:
        # Fallback
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_mask = mask
        best_size = len(best_subset)

    return np.array(best_subset, dtype=np.int64)
