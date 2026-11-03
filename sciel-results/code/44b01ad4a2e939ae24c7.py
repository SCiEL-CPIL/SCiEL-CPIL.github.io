import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            tset_to_blocks[idx].append(i)
        block_masks[i] = mask

    full_mask = (1 << num_tsets) - 1

    # Helper: Check if a set of blocks covers everything
    def is_valid_solution(blocks):
        covered = 0
        for b in blocks:
            covered |= block_masks[b]
        return covered == full_mask

    # Helper: Get cover counts
    def get_cover_counts(blocks):
        counts = [0] * num_tsets
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # 2. Greedy
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            # Randomize order to break ties
            # We can't shuffle the whole list every time.
            # We can pick a random subset of blocks to check if we are in a hurry?
            # No, we need the best.
            
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    # Optional: Random tie breaking
                    # if count == best_count and rng.random() < 0.5:
                    #     best_idx = i # This is wrong, we need to pick one of the maxes
                    # Better: collect all maxes? Too slow.
                    # Just take the first max found.
            
            if best_idx == -1 or best_count == 0:
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = get_cover_counts(sol)
        
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
                if pos >= len(sol): continue
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
                    # Update counts
                    m = block_masks[b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
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
                
            # Phase 2: Swap
            if sol:
                found_swap = False
                attempts = 20
                if elapsed() < deadline - 30: attempts = 50
                
                for _ in range(attempts):
                    if not sol: break
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
                        m = block_masks[b_out]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list: continue
                    
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                            
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]: continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)
                            
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Update counts
                        m = block_masks[b_out]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        m = block_masks[b_in]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] += 1
                            m ^= lsb
                        in_sol[b_out] = False
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

    # 4. Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # Initial run
    sol = greedy_cover(0)
    ls_sol = local_search(sol, 0)
    best_solution = ls_sol
    best_size = len(ls_sol)
    
    # ILS Loop
    current_sol = best_solution
    while elapsed() < deadline - 20:
        # Perturb
        # Remove 3-5 random blocks
        # Then add random blocks to fix coverage?
        # Or just add random blocks until valid?
        
        perturbed = list(current_sol)
        rng = random.Random(int(elapsed() * 1000))
        
        # Remove random blocks
        num_remove = min(3, len(perturbed))
        for _ in range(num_remove):
            if not perturbed: break
            idx = rng.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Fix coverage by adding blocks
        # Check what is uncovered
        covered = 0
        for b in perturbed:
            covered |= block_masks[b]
        uncovered = full_mask & ~covered
        
        # Add blocks greedily to fix
        while uncovered:
            if elapsed() > deadline - 10:
                break
            # Find block covering most uncovered
            best_idx = -1
            best_count = -1
            # To speed up, only check a subset?
            # Or check all.
            # This fix phase might be slow if we do it many times.
            # Let's limit the fix phase.
            
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    
            if best_idx == -1 or best_count == 0:
                break
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        # Local Search on Perturbed
        ls_sol = local_search(perturbed, int(elapsed() * 1000))
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            current_sol = best_solution
        else:
            # Keep current_sol as base for next perturbation?
            # Or restart from best?
            # Standard ILS keeps the best and perturbs it.
            pass

    return [list(all_blocks[i]) for i in best_solution]
