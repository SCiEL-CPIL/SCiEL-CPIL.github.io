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
            
            # Optimization: Randomize start index to break ties and avoid bias
            start_i = rng.randint(0, num_blocks - 1)
            
            for i in range(num_blocks):
                idx = (start_i + i) % num_blocks
                count = (block_masks[idx] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    if best_count == num_tsets: # Can't get better
                        break
            
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
        max_no_improve = 20 # Reduced to allow more ILS iterations
        
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
                # Try multiple candidates for swap
                # We try to remove a block and find a replacement
                # To speed up, we limit the number of attempts
                attempts = 10
                if elapsed() < deadline - 30: attempts = 30
                
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
                    
                    # Sort critical tsets by number of blocks covering them (ascending)
                    # to prioritize the most constrained ones
                    crit_ts_list.sort(key=lambda x: len(tset_to_blocks[x]))
                    
                    # Try the most constrained critical tset first
                    min_ts_idx = crit_ts_list[0]
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
    seed_counter = 1
    
    while elapsed() < deadline - 20:
        # Perturb
        perturbed = list(current_sol)
        rng = random.Random(seed_counter)
        
        # Remove 2-4 random blocks
        num_remove = rng.randint(2, 4)
        num_remove = min(num_remove, len(perturbed))
        for _ in range(num_remove):
            if not perturbed: break
            idx = rng.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Fix coverage by adding blocks greedily
        covered = 0
        for b in perturbed:
            covered |= block_masks[b]
        uncovered = full_mask & ~covered
        
        # Add blocks greedily to fix
        while uncovered:
            if elapsed() > deadline - 10:
                break
            best_idx = -1
            best_count = -1
            # Random start for tie breaking
            start_i = rng.randint(0, num_blocks - 1)
            for i in range(num_blocks):
                idx = (start_i + i) % num_blocks
                count = (block_masks[idx] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    if best_count == num_tsets:
                        break
                    
            if best_idx == -1 or best_count == 0:
                break
            perturbed.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        # Local Search on Perturbed
        ls_sol = local_search(perturbed, seed_counter)
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            current_sol = best_solution
        else:
            # Occasionally restart from a fresh greedy solution if stuck
            if rng.random() < 0.1:
                fresh_sol = greedy_cover(seed_counter)
                fresh_ls = local_search(fresh_sol, seed_counter + 100)
                if len(fresh_ls) < best_size:
                    best_size = len(fresh_ls)
                    best_solution = fresh_ls
                    current_sol = best_solution
            
        seed_counter += 1

    return [list(all_blocks[i]) for i in best_solution]
