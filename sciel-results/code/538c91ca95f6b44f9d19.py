import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0

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

    # 2. Greedy with randomized tie-breaking
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_count = -1
            best_candidates = []
            
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_candidates = [i]
                elif count == best_count and count > 0:
                    best_candidates.append(i)
            
            if not best_candidates or best_count == 0:
                break
                
            best_idx = rng.choice(best_candidates)
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Enhanced Local Search with 2-opt moves
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
        max_no_improve = 30
        
        while elapsed() < deadline - 8 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant blocks
            if sol:
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
                
            # Phase 2: 1-opt swap (remove 1, add 1)
            if sol:
                found_swap = False
                num_attempts = 30 if elapsed() < deadline - 25 else 15
                
                for _ in range(num_attempts):
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
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
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
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        break
                        
                if found_swap:
                    no_improve = 0
                    continue
            
            # Phase 3: 2-opt move (remove 1, add 2) - try to find better structure
            if sol and elapsed() < deadline - 15:
                found_2opt = False
                num_2opt_attempts = 10
                
                for _ in range(num_2opt_attempts):
                    if not sol: break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    # What would be uncovered if we remove b_out?
                    potential_uncovered = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            potential_uncovered |= lsb
                        m ^= lsb
                    
                    if potential_uncovered == 0:
                        # Just remove it
                        m = block_masks[b_out]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_2opt = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        break
                    
                    # Find two blocks that together cover potential_uncovered
                    # Try pairs from the most constrained tset
                    crit_ts_list = []
                    m = potential_uncovered
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
                    
                    candidate_blocks = [b for b in tset_to_blocks[min_ts_idx] if not in_sol[b]]
                    
                    if len(candidate_blocks) >= 2:
                        # Try random pairs
                        for _ in range(5):
                            if len(candidate_blocks) < 2: break
                            i1, i2 = rng.sample(range(len(candidate_blocks)), 2)
                            b1, b2 = candidate_blocks[i1], candidate_blocks[i2]
                            
                            # Check if b1 and b2 together cover potential_uncovered
                            if (block_masks[b1] | block_masks[b2]) & potential_uncovered == potential_uncovered:
                                # Apply the 2-opt move
                                m = block_masks[b_out]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cover_counts[idx] -= 1
                                    m ^= lsb
                                m = block_masks[b1]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cover_counts[idx] += 1
                                    m ^= lsb
                                m = block_masks[b2]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cover_counts[idx] += 1
                                    m ^= lsb
                                in_sol[b_out] = False
                                in_sol[b1] = True
                                in_sol[b2] = True
                                sol.pop(pos)
                                sol.extend([b1, b2])
                                improved = True
                                found_2opt = True
                                break
                            
                if found_2opt:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            
            no_improve += 1
        
        return best_sol

    # 4. Iterated Local Search with stronger perturbation
    best_solution = None
    best_size = float('inf')
    
    # Multiple initial greedy runs
    for seed in range(5):
        if elapsed() > deadline - 30:
            break
        sol = greedy_cover(seed)
        ls_sol = local_search(sol, seed)
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
    
    if best_solution is None:
        best_solution = greedy_cover(0)
        best_size = len(best_solution)
    
    # ILS Loop
    current_sol = best_solution
    ills_seed = 100
    
    while elapsed() < deadline - 15:
        # Perturb with stronger moves
        perturbed = list(current_sol)
        rng = random.Random(ills_seed)
        ills_seed += 1
        
        # Remove 2-4 random blocks
        num_remove = min(2 + rng.randint(0, 2), len(perturbed))
        for _ in range(num_remove):
            if not perturbed: break
            idx = rng.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Fix coverage by adding blocks greedily
        covered = 0
        for b in perturbed:
            covered |= block_masks[b]
        uncovered = full_mask & ~covered
        
        while uncovered:
            if elapsed() > deadline - 8:
                break
            best_idx = -1
            best_count = -1
            best_candidates = []
            
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_candidates = [i]
                elif count == best_count and count > 0:
                    best_candidates.append(i)
                    
            if not best_candidates or best_count == 0:
                break
            best_idx = rng.choice(best_candidates)
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        # Local Search on Perturbed
        ls_sol = local_search(perturbed, ills_seed)
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            current_sol = best_solution
    
    return [list(all_blocks[i]) for i in best_solution]
