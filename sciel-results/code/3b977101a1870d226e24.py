import itertools
import time
import random
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0 # Safety margin

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

    # 2. Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            # To speed up, we can sort blocks initially by size (all same) or random shuffle
            # But we need to pick the best one each time.
            
            # Optimization: Keep a list of candidate indices.
            # If a block is selected, remove it.
            # Scanning all blocks each time is O(N_blocks * N_iterations).
            # 75k * 300 = 22.5M. Acceptable.
            
            # We will iterate through all blocks and pick the one with max new coverage.
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count == num_tsets: # Perfect cover (unlikely)
                        break
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search (1-for-1 swap and deletion)
    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
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
        
        # Simulated Annealing parameters
        T_start = 1.0
        T_end = 0.01
        sa_steps = 1000 # Total steps for SA phase if triggered
        
        # We integrate SA into the main loop.
        # If we are stuck (no_improve > 0), we might do some SA steps.
        # Or we can just run a dedicated SA phase after standard local search fails to improve.
        
        # Let's modify the loop to include SA when stuck.
        current_T = T_start
        sa_counter = 0
        
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
                # Reset SA temperature if we made a significant improvement?
                # Or just keep cooling? Let's keep cooling but maybe reset if we found a much better solution.
                continue
                
            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                # Increase swap attempts if we have time
                max_swap_attempts = 20
                if elapsed() < deadline - 30:
                    max_swap_attempts = 50
                    
                for _ in range(max_swap_attempts):
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
                    # To find candidates, intersect tset_to_blocks for all tsets in critical_mask
                    # This is expensive if critical_mask has many bits.
                    # Average bits in critical_mask?
                    # Block has 56 tsets. If cover is dense, critical is small.
                    
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
                        
                    # Optimization: Find the critical tset with the fewest covering blocks
                    # This minimizes the candidate pool we need to filter
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                            
                    # Filter candidates from the shortest list
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        # Check if this block covers ALL critical tsets
                        # (block_masks[b_idx] & critical_mask) == critical_mask
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)
                            
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Perform swap
                        # Calculate change in "energy" (size)
                        # Since we are swapping 1 for 1, size doesn't change.
                        # Wait, standard local search only keeps swaps if they don't increase size?
                        # No, 1-for-1 swap keeps size same.
                        # The problem is that we might need to accept a swap that *temporarily* increases size?
                        # No, 1-for-1 swap keeps size constant.
                        # The "Simulated Annealing" idea usually applies to changes that can increase cost.
                        # In covering design, we can also do "2-for-1" (add 1 remove 2 -> size -1) or "1-for-2" (add 2 remove 1 -> size +1).
                        # The current code only does 1-for-1.
                        # If we are stuck, it means no 1-for-1 swap improves the solution (which is impossible if size is fixed, unless we define "improve" as leading to a future deletion).
                        # Actually, 1-for-1 swaps don't change size. They just move us in the solution space.
                        # The "improvement" in Phase 1 is deletion.
                        # So we need to find a 1-for-1 swap that makes a deletion possible.
                        # The current code picks a random valid candidate.
                        # If we pick a candidate that doesn't enable deletion, we are stuck.
                        # SA can help by allowing "worse" states? But 1-for-1 is neutral.
                        # Maybe we should allow 1-for-2 swaps (size +1) to escape local minima where no 1-for-1 or deletion works?
                        # Yes, that's the standard SA approach for set cover: allow adding blocks temporarily.
                        
                        # Let's implement 1-for-2 swaps (Add 1, Remove 2? No, Add 2, Remove 1 -> Size +1)
                        # Or Add 1, Remove 0 -> Size +1 (just adding a block).
                        # Let's try adding a block (Size +1) with SA.
                        
                        # For now, let's just do the 1-for-1 swap as before, but we need to be smarter about which one to pick.
                        # The current code picks randomly.
                        # Let's try to pick the one that maximizes "potential" for deletion?
                        # That's hard to calculate.
                        
                        # Let's stick to the plan: Implement SA with Add/Remove operations.
                        # The current loop only does 1-for-1.
                        # I will add a "SA Phase" that allows adding blocks.
                        
                        # Perform swap
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True # Mark as improved to reset no_improve counter?
                        # Actually, 1-for-1 swap doesn't reduce size.
                        # It only "improves" if it leads to a deletion.
                        # So we shouldn't reset no_improve immediately.
                        # But we did change the state.
                        # Let's break out of the swap loop to try deletions again.
                        break
                        
                if found_swap:
                    # Don't update best_sol unless size decreased
                    # no_improve stays same? Or reset?
                    # If we swapped, we might be able to delete next.
                    # Let's not reset no_improve for neutral swaps.
                    continue
            else:
                break
                
            # If no deletion and no swap found (or swap didn't help immediately)
            no_improve += 1
            
            # SA Phase: If stuck, try adding a block
            if no_improve > 5 and sa_counter < sa_steps:
                # Temperature decay
                # T = T_start * (T_end / T_start) ** (sa_counter / sa_steps)
                # Simpler: T = T_start * 0.99 ** sa_counter
                current_T = T_start * (0.99 ** sa_counter)
                sa_counter += 1
                
                # Try to add a random block that covers some uncovered tsets?
                # Or just a random block not in solution?
                # Adding a block increases size by 1.
                # Probability of accepting: exp(-1 / T)
                # Since size increases by 1, Delta E = 1.
                
                # Find a block to add
                # Prefer blocks that cover many tsets currently covered by only 1 block?
                # Or just random.
                
                # Let's pick a random block not in solution
                # This is expensive if we scan all blocks.
                # We can maintain a list of non-solution blocks?
                # Or just pick a random index and check.
                
                # Let's try to add a block that helps with "critical" tsets of some existing block?
                # Actually, let's just add a random block.
                
                # To make it more effective, let's pick a block that covers at least one tset with count == 1?
                # No, that would make it redundant immediately? No, adding it increases count to 2.
                # It doesn't help with deletion of the other block unless we remove the other one.
                
                # Let's just add a random block.
                b_add = rng.randint(0, num_blocks - 1)
                if not in_sol[b_add]:
                    # Accept with probability exp(-1/T)
                    if rng.random() < math.exp(-1.0 / current_T):
                        # Add it
                        update_counts(b_add, 1)
                        in_sol[b_add] = True
                        sol.append(b_add)
                        # Don't update best_sol (size increased)
                        # Reset no_improve? Maybe.
                        no_improve = 0 # Give it a chance to find a path to improvement
                        continue # Try deletions/swaps again
                
        # Final Polish: Ensure minimal
        # Run deletion pass one last time
        changed = True
        while changed:
            changed = False
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
                    changed = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
        
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run greedy restarts
    # Adjust number of restarts based on time taken for first one
    num_restarts = 5
    for seed in range(num_restarts):
        if elapsed() > deadline - 20:
            break
            
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    if best_solution is None:
        # Fallback: return empty or raise
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
