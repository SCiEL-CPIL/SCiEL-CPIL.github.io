import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0 # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks and their bitmasks
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        all_blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Initial full mask
    full_mask = (1 << num_tsets) - 1

    # 2. Greedy Set Cover (Fixed: removed redundant allocations inside loop)
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        # Pre-allocate used array outside the loop
        used = [False] * num_blocks
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count == num_tsets: # Perfect cover (unlikely)
                        break
            
            if best_idx == -1 or best_count == 0:
                break
                
            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    # 3. Local Search (Fixed: efficient swap logic, no set conversions)
    def local_search_v2(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                m = block_masks[b]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                for _ in range(20): # Try more swaps
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                        
                    if critical_mask == 0:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    # Find candidate B
                    # Must cover critical_mask
                    # Must not be in_sol
                    # Efficiently find candidates by intersecting lists via a counter array
                    
                    # Get list of tsets in critical_mask
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list:
                        continue
                        
                    # Use a dictionary or list to count occurrences in candidate lists
                    # Since num_blocks is ~75k, a list of size num_blocks is okay if we clear it fast
                    # Or just use a set of candidates from the first list and filter
                    # Let's stick to the set intersection but optimize the removal of in_sol
                    # Actually, the previous code did:
                    # candidates = set(tset_to_blocks[crit_ts_list[0]])
                    # for ts_idx in crit_ts_list[1:]:
                    #     candidates &= set(tset_to_blocks[ts_idx])
                    # candidates -= set(sol)
                    
                    # Optimization:
                    # 1. Start with the smallest list among crit_ts_list to minimize initial set size?
                    #    tset_to_blocks lists are roughly same size.
                    # 2. Instead of set(sol), iterate candidates and check in_sol[c]
                    
                    # Let's implement a fast intersection
                    # If crit_ts_list is small (which it should be for dense covers),
                    # we can just take the first list, and for each element, check if it's in all others.
                    # But "in all others" is expensive if we do linear scan.
                    # Set intersection is O(N) per set.
                    
                    # Let's try a different approach for swap:
                    # We want to find a block B not in sol that covers critical_mask.
                    # We can iterate through all blocks? No, 75k is too slow for 20*50 iterations.
                    
                    # Let's optimize the set intersection:
                    # candidates = set(tset_to_blocks[crit_ts_list[0]])
                    # for ts_idx in crit_ts_list[1:]:
                    #     candidates.intersection_update(tset_to_blocks[ts_idx])
                    # This avoids creating a new set for tset_to_blocks[ts_idx] if it's a list?
                    # intersection_update accepts any iterable.
                    
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])
                        
                    # Filter out blocks already in sol
                    # Instead of candidates -= set(sol), we can just check in_sol when picking
                    # But we need to pick one.
                    # If candidates is empty, continue.
                    # If not, pick a random one that is not in_sol.
                    # Since candidates is small, we can just list it and filter.
                    
                    valid_candidates = [c for c in candidates if not in_sol[c]]
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Perform swap
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run more greedy restarts since greedy is faster now
    # Previous: 5 restarts
    # New: Increase restarts if time permits, or just let the loop run until deadline
    # But local_search is the heavy part.
    # Let's run 5 restarts as before, but the fixed greedy is faster, so we have more time for local search.
    
    for seed in range(5):
        if elapsed() > deadline - 30:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    # If we have time left, try more seeds?
    # The deadline guard in local_search handles the rest.
    # But if local_search finishes early, we can try another seed.
    
    # Let's add a loop to use remaining time
    seed = 5
    while elapsed() < deadline - 10:
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
        seed += 1

    if best_solution is None:
        # Fallback: return empty or raise
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
