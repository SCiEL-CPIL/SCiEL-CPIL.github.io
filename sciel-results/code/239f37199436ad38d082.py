import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each block, compute its bitmask of covered t-subsets
    # Also store the list of indices for quick access if needed, but bitmask is primary
    block_masks = [0] * num_k
    block_t_indices = [[] for _ in range(num_k)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        indices = []
        for combo in itertools.combinations(block, t):
            idx = t_index[combo]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    # Build reverse mapping: t-subset index -> list of block indices
    t_to_blocks = [[] for _ in range(num_t)]
    for i in range(num_k):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)

    # Initial full mask
    full_mask = (1 << num_t) - 1

    def run_greedy(seed=42):
        """
        Greedy set cover using bitmasks for fast gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_mask
        selected = []
        selected_set = set()
        
        # We will re-evaluate gains dynamically. 
        # To speed this up, we can just iterate all blocks each time, 
        # but num_k is ~75k, which might be slow in Python.
        # However, bit_count is C-speed. 
        # Let's try to optimize by only considering blocks that cover at least one uncovered set.
        # A block is "active" if (block_mask & uncovered_mask) != 0.
        
        # Simple approach: Iterate all, pick best.
        # If too slow, we can maintain a priority queue, but updates are expensive.
        # Given num_k ~ 75k and num_t ~ 11k, and greedy usually picks ~300-400 blocks,
        # 300 * 75k = 22.5M operations. bit_count is fast. This should be okay.
        
        while uncovered_mask:
            if time_left() < 20:
                break
            
            best_idx = -1
            best_gain = -1
            
            # Optimization: Skip blocks already selected
            for i in range(num_k):
                if i in selected_set:
                    continue
                m = block_masks[i]
                if not (m & uncovered_mask):
                    continue
                gain = (m & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        return selected

    def local_search(selected, seed=42, max_time_frac=0.15):
        """
        Local search with deletion, 1-for-1 swap, and simulated annealing.
        Uses bitmasks and cover counts.
        """
        if not selected:
            return selected
        
        rng = random.Random(seed)
        sol = list(selected)
        sol_set = set(sol)
        
        # Initialize cover counts
        cover_counts = np.zeros(num_t, dtype=np.int32)
        for b in sol:
            for idx in block_t_indices[b]:
                cover_counts[idx] += 1
        
        # Initialize single_cover_mask: bit i is 1 if cover_counts[i] == 1
        single_mask = 0
        for i in range(num_t):
            if cover_counts[i] == 1:
                single_mask |= (1 << i)
                
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        temperature = 2.0
        cooling = 0.98

        def apply_move_remove(block_idx):
            nonlocal cover_counts, single_mask, sol, sol_set
            for idx in block_t_indices[block_idx]:
                cover_counts[idx] -= 1
                if cover_counts[idx] == 1:
                    single_mask |= (1 << idx)
                elif cover_counts[idx] == 0:
                    # This shouldn't happen in valid moves, but handle it
                    single_mask &= ~(1 << idx)
                    # Actually if it becomes 0, it's no longer single.
                    # If it was 1, it becomes 0.
                    # If it was >1, it becomes >0.
                    # Logic:
                    # old_count = cover_counts[idx] + 1
                    # new_count = cover_counts[idx]
                    # if old_count == 1: new is 0. single_mask bit was 1, now 0.
                    # if old_count > 1: new is >0. single_mask bit was 0, now 0 (unless new==1, but old>1 implies new>=1, if old=2 new=1, single_mask bit becomes 1).
                    pass
            
            # Correct single_mask update logic:
            # We need to recalculate bits for affected indices.
            # For efficiency, let's just update bits for indices in block.
            # old_count = cover_counts[idx] + 1
            # new_count = cover_counts[idx]
            # if old_count == 1: bit was 1, now 0.
            # if old_count == 2: bit was 0, now 1.
            # else: bit stays 0.
            
            # Let's do it properly in the loop above.
            for idx in block_t_indices[block_idx]:
                new_c = cover_counts[idx]
                old_c = new_c + 1
                if old_c == 1:
                    single_mask &= ~(1 << idx)
                elif old_c == 2:
                    single_mask |= (1 << idx)
                # else no change to single status
            
            sol_set.discard(block_idx)
            sol.remove(block_idx)

        def apply_move_add(block_idx):
            nonlocal cover_counts, single_mask, sol, sol_set
            for idx in block_t_indices[block_idx]:
                old_c = cover_counts[idx]
                cover_counts[idx] += 1
                new_c = cover_counts[idx]
                if old_c == 0:
                    # Becomes 1, so single
                    single_mask |= (1 << idx)
                elif old_c == 1:
                    # Becomes 2, so no longer single
                    single_mask &= ~(1 << idx)
            
            sol_set.add(block_idx)
            sol.append(block_idx)

        def apply_move_swap(out_idx, in_idx):
            nonlocal cover_counts, single_mask, sol, sol_set
            # Remove out
            for idx in block_t_indices[out_idx]:
                old_c = cover_counts[idx]
                cover_counts[idx] -= 1
                new_c = cover_counts[idx]
                if old_c == 1:
                    single_mask &= ~(1 << idx)
                elif old_c == 2:
                    single_mask |= (1 << idx)
            
            sol_set.discard(out_idx)
            sol.remove(out_idx)
            
            # Add in
            for idx in block_t_indices[in_idx]:
                old_c = cover_counts[idx]
                cover_counts[idx] += 1
                new_c = cover_counts[idx]
                if old_c == 0:
                    single_mask |= (1 << idx)
                elif old_c == 1:
                    single_mask &= ~(1 << idx)
            
            sol_set.add(in_idx)
            sol.append(in_idx)

        # SA Loop
        while time_left() > 15 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Try to remove redundant blocks
            if sol:
                order = list(range(len(sol)))
                rng.shuffle(order)
                for pos in order:
                    if pos >= len(sol):
                        continue
                    b = sol[pos]
                    # Check if any t-subset in b is only covered by b
                    # i.e., (block_mask[b] & single_mask) == 0
                    if (block_masks[b] & single_mask) == 0:
                        apply_move_remove(b)
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
                for _ in range(20): # Try a few random swaps
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    # Critical sets: t-subsets covered only by b_out
                    critical_mask = block_masks[b_out] & single_mask
                    if critical_mask == 0:
                        # Should have been caught in Phase 1, but handle just in case
                        apply_move_remove(b_out)
                        improved = True
                        found_swap = True
                        break
                    
                    # Find candidates that cover all critical sets
                    # Get list of critical indices
                    crit_indices = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_indices.append(idx)
                        m ^= lsb
                    
                    if not crit_indices:
                        continue
                        
                    # Intersect candidate lists
                    candidates = set(t_to_blocks[crit_indices[0]])
                    for c_idx in crit_indices[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(t_to_blocks[c_idx])
                    
                    candidates -= sol_set
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        # Check if this swap is valid (it should be by construction, 
                        # but we must ensure no new uncovered sets are created? 
                        # No, swap maintains size. 
                        # We need to ensure that removing b_out doesn't uncover anything 
                        # that b_in doesn't cover. 
                        # Since b_in covers all critical sets of b_out, 
                        # and all other sets in b_out have count >= 2, 
                        # removing b_out leaves them >= 1. 
                        # Adding b_in only increases counts. 
                        # So it is always valid.)
                        
                        apply_move_swap(b_out, b_in)
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue

            # Phase 3: Simulated Annealing Move (2-for-1 or 1-for-2)
            # Try to reduce size by 1 (2-for-1)
            if len(sol) >= 2 and rng.random() < 0.7:
                i, j = rng.sample(range(len(sol)), 2)
                b1, b2 = sol[i], sol[j]
                
                # Check if removing both uncovers anything
                # Temporarily remove b1, b2
                temp_cover = cover_counts.copy()
                for idx in block_t_indices[b1]:
                    temp_cover[idx] -= 1
                for idx in block_t_indices[b2]:
                    temp_cover[idx] -= 1
                
                # Find uncovered
                uncovered_mask = 0
                for idx in range(num_t):
                    if temp_cover[idx] == 0:
                        uncovered_mask |= (1 << idx)
                        
                if uncovered_mask == 0:
                    # Both can be removed!
                    apply_move_remove(b1)
                    # b2 index changed? 
                    # apply_move_remove modifies sol. 
                    # We need to be careful with indices.
                    # Let's just rebuild or do it manually.
                    # Actually, apply_move_remove removes from sol list.
                    # If we remove b1, b2 might shift.
                    # Let's do manual update for 2-for-0
                    # Roll back temp_cover
                    # Just do it directly:
                    for idx in block_t_indices[b1]:
                        old_c = cover_counts[idx]
                        cover_counts[idx] -= 1
                        new_c = cover_counts[idx]
                        if old_c == 1: single_mask &= ~(1 << idx)
                        elif old_c == 2: single_mask |= (1 << idx)
                    for idx in block_t_indices[b2]:
                        old_c = cover_counts[idx]
                        cover_counts[idx] -= 1
                        new_c = cover_counts[idx]
                        if old_c == 1: single_mask &= ~(1 << idx)
                        elif old_c == 2: single_mask |= (1 << idx)
                    
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    sol.remove(b1)
                    sol.remove(b2)
                    
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue

                if uncovered_mask:
                    # Find a block that covers all uncovered
                    unc_indices = []
                    m = uncovered_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        unc_indices.append(idx)
                        m ^= lsb
                        
                    candidates = set(t_to_blocks[unc_indices[0]])
                    for u_idx in unc_indices[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(t_to_blocks[u_idx])
                    candidates -= sol_set
                    
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        # Check if b_in covers all uncovered
                        # It should by construction.
                        # Apply 2-for-1
                        for idx in block_t_indices[b1]:
                            old_c = cover_counts[idx]
                            cover_counts[idx] -= 1
                            new_c = cover_counts[idx]
                            if old_c == 1: single_mask &= ~(1 << idx)
                            elif old_c == 2: single_mask |= (1 << idx)
                        for idx in block_t_indices[b2]:
                            old_c = cover_counts[idx]
                            cover_counts[idx] -= 1
                            new_c = cover_counts[idx]
                            if old_c == 1: single_mask &= ~(1 << idx)
                            elif old_c == 2: single_mask |= (1 << idx)
                        
                        for idx in block_t_indices[b_in]:
                            old_c = cover_counts[idx]
                            cover_counts[idx] += 1
                            new_c = cover_counts[idx]
                            if old_c == 0: single_mask |= (1 << idx)
                            elif old_c == 1: single_mask &= ~(1 << idx)
                            
                        sol_set.discard(b1)
                        sol_set.discard(b2)
                        sol.remove(b1)
                        sol.remove(b2)
                        sol_set.add(b_in)
                        sol.append(b_in)
                        
                        # Accept if improves or with probability
                        # Size decreases by 1, so always accept if valid
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        no_improve = 0
                        continue

            no_improve += 1
            temperature *= cooling
            if temperature < 0.01:
                break

        return best_sol

    # Run multiple strategies
    best_selected = None
    best_len = float('inf')

    configs = [
        42,
        123,
        777,
        555,
        2024,
    ]

    for seed in configs:
        if time_left() < 30:
            break
        sel = run_greedy(seed)
        sel = local_search(sel, seed=seed)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]

    # Final polish
    if best_selected and time_left() > 20:
        best_selected = local_search(best_selected, seed=999)

    result = [list(k_blocks[i]) for i in best_selected]

    # Ensure validity
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in best_selected:
            for idx in block_t_indices[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            selected_set = set(best_selected)
            for i in range(num_k):
                if time_left() < 3:
                    break
                if i in selected_set:
                    continue
                for idx in block_t_indices[i]:
                    if final_cover[idx] == 0:
                        best_selected.append(i)
                        selected_set.add(i)
                        for idx2 in block_t_indices[i]:
                            final_cover[idx2] += 1
                        break
            result = [list(k_blocks[i]) for i in best_selected]
            
    return result
