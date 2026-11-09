<<<<<<< SEARCH
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            best_set = []
        elif not check_divisibility(non_empty):
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        return [256]
    
    # Convert to pure Python ints
    return [int(x) for x in best_set]
=======
    # New Strategy: Deep Search for larger sets using high-magnitude candidates
    # Try to find sets of size 8, 9, 10 using DFS with a restricted candidate pool
    # to avoid combinatorial explosion.
    
    # Candidate pool: Numbers > 100 are less likely to divide each other's sums
    # because the sums are large and the ratios are closer to 1.
    high_candidates = list(range(256, 100, -1))
    
    for target_size in range(best_size + 1, 12):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Shuffle candidates to try different branches
        random.shuffle(high_candidates)
        # Limit pool size to keep DFS manageable
        pool = high_candidates[:50]
        
        found_set = deep_search(target_size, pool)
        if found_set:
            # Verify
            sums_found = compute_sums(found_set)
            if check_divisibility(sums_found - {0}):
                best_size = len(found_set)
                best_set = found_set[:]
                best_sums = sums_found
                # Update local search with this new best
                S_ls4, sums_ls4 = local_search(best_set, best_sums)
                if len(S_ls4) > best_size:
                    best_size = len(S_ls4)
                    best_set = S_ls4[:]
                    best_sums = sums_ls4
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            best_set = []
        elif not check_divisibility(non_empty):
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        return [256]
    
    # Convert to pure Python ints
    return [int(x) for x in best_set]
>>>>>>> REPLACE
