def greedy_once(v, k, t, all_5_subsets, seed):
    random.seed(seed)
    uncovered_set = set(all_5_subsets)
    uncovered_list = list(all_5_subsets)
    blocks = []
    
    # Shuffle the list initially to ensure randomness in the "first" pick if we were doing that,
    # but we are picking randomly from the set.
    # Actually, picking random from list and checking set is faster.
    
    while uncovered_set:
        # Pick a random uncovered target
        target = None
        attempts = 0
        while target is None:
            if attempts > 100: # Safety
                # Fallback: find one
                for s in uncovered_list:
                    if s in uncovered_set:
                        target = s
                        break
                break
            idx = random.randint(0, len(uncovered_list) - 1)
            candidate = uncovered_list[idx]
            if candidate in uncovered_set:
                target = candidate
            attempts += 1
            
        # Generate candidates
        remaining_points = [p for p in range(v) if p not in target]
        best_count = -1
        best_candidates = []
        
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target + comb))
            count = 0
            for s5 in itertools.combinations(block, t):
                if s5 in uncovered_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidates = [block]
            elif count == best_count:
                best_candidates.append(block)
                
        chosen_block = random.choice(best_candidates)
        blocks.append(list(chosen_block))
        
        # Update uncovered_set
        for s5 in itertools.combinations(chosen_block, t):
            uncovered_set.discard(s5)
            
    return blocks
