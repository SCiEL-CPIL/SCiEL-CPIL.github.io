import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety

    # Precompute all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(k_subs)
    
    # Precompute block_t_subs: for each block, list of t-subset indices it covers
    # And t_sub_blocks: for each t-subset, list of block indices that cover it
    block_t_subs = [[] for _ in range(n_k_subs)]
    t_sub_blocks = [[] for _ in range(n_t_subs)]
    
    for b_idx, k_sub in enumerate(k_subs):
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_to_idx[t_sub]
            block_t_subs[b_idx].append(t_idx)
            t_sub_blocks[t_idx].append(b_idx)
            
    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Greedy Construction using score updates
    def greedy_construct():
        if time_left() < 1:
            return None
        
        # block_scores[i] = number of currently uncovered t-subsets covered by block i
        block_scores = [0] * n_k_subs
        # Initialize scores: each block covers C(k,t) t-subsets
        # But we only care about uncovered ones. Initially all are uncovered.
        for i in range(n_k_subs):
            block_scores[i] = len(block_t_subs[i])
            
        uncovered = set(range(n_t_subs))
        selected_blocks = []
        selected_set = set()
        
        while uncovered and time_left() > 5:
            # Find block with max score
            best_idx = -1
            best_score = -1
            # Optimization: We can't just scan all 75k blocks every time if it's slow.
            # But 75k is small. 330 iterations * 75k = 24M checks. Fast.
            for i in range(n_k_subs):
                if i in selected_set:
                    continue
                if block_scores[i] > best_score:
                    best_score = block_scores[i]
                    best_idx = i
            
            if best_idx == -1:
                # No block covers any uncovered subset? Should not happen if covering exists.
                break
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update scores for blocks that share t-subsets with best_idx
            for t_idx in block_t_subs[best_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    # Decrement score of other blocks covering this t_idx
                    for other_b in t_sub_blocks[t_idx]:
                        if other_b != best_idx and other_b not in selected_set:
                            block_scores[other_b] -= 1
                            
        return selected_blocks

    # Local Search: Reduction + 1-Swap
    def local_search(initial_blocks):
        if time_left() < 5:
            return initial_blocks
            
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Compute cover counts
        cover_count = [0] * n_t_subs
        for b in current_blocks:
            for t_idx in block_t_subs[b]:
                cover_count[t_idx] += 1
                
        def reduce_solution():
            if time_left() < 2:
                return
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in current_blocks:
                    is_redundant = True
                    for t_idx in block_t_subs[b]:
                        if cover_count[t_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove b
                        for t_idx in block_t_subs[b]:
                            cover_count[t_idx] -= 1
                        changed = True
                    else:
                        new_blocks.append(b)
                if changed:
                    current_blocks = new_blocks
                    current_set = set(current_blocks)
                    
        reduce_solution()
        
        # 1-Swap Phase
        # Try to swap blocks to enable further reduction
        # We iterate multiple times
        max_swap_iterations = 50
        for _ in range(max_swap_iterations):
            if time_left() < 5:
                break
                
            improved = False
            # Shuffle to avoid bias
            blocks_to_try = list(current_blocks)
            random.shuffle(blocks_to_try)
            
            for b_out in blocks_to_try:
                if time_left() < 2:
                    break
                if b_out not in current_set:
                    continue
                    
                # Find t-subsets that are ONLY covered by b_out
                critical_ts = []
                for t_idx in block_t_subs[b_out]:
                    if cover_count[t_idx] == 1:
                        critical_ts.append(t_idx)
                        
                if not critical_ts:
                    # Redundant, should have been removed
                    continue
                    
                # Find candidate blocks b_in that cover ALL critical_ts
                # and are not in current_set
                # To be efficient, we can intersect the lists of blocks covering each critical_t
                # Start with blocks covering the first critical_t
                candidates = set(t_sub_blocks[critical_ts[0]])
                candidates -= current_set
                
                for t_idx in critical_ts[1:]:
                    candidates &= set(t_sub_blocks[t_idx])
                    if not candidates:
                        break
                        
                if candidates:
                    # Pick a random candidate
                    b_in = random.choice(list(candidates))
                    
                    # Perform swap
                    # Remove b_out
                    for t_idx in block_t_subs[b_out]:
                        cover_count[t_idx] -= 1
                    current_set.remove(b_out)
                    current_blocks.remove(b_out)
                    
                    # Add b_in
                    for t_idx in block_t_subs[b_in]:
                        cover_count[t_idx] += 1
                    current_set.add(b_in)
                    current_blocks.append(b_in)
                    
                    # Try to reduce
                    old_len = len(current_blocks)
                    reduce_solution()
                    
                    if len(current_blocks) < old_len:
                        improved = True
                        # Keep the change
                    else:
                        # Revert
                        # Remove b_in
                        for t_idx in block_t_subs[b_in]:
                            cover_count[t_idx] -= 1
                        current_set.remove(b_in)
                        current_blocks.remove(b_in)
                        
                        # Add b_out
                        for t_idx in block_t_subs[b_out]:
                            cover_count[t_idx] += 1
                        current_set.add(b_out)
                        current_blocks.append(b_out)
                        
                        # Note: Reduction might have changed things, but we reverted the swap.
                        # The state of cover_count is now consistent with current_blocks.
                        # However, reduce_solution might have removed blocks.
                        # If we revert, we need to ensure we didn't lose blocks removed by reduction.
                        # This logic is tricky. 
                        # Simpler: Only accept swap if it leads to a valid state. 
                        # Reduction is only done if we accept?
                        # Let's change strategy: 
                        # 1. Swap.
                        # 2. Check if valid (it is by construction of critical_ts).
                        # 3. Try to reduce.
                        # 4. If reduction improves size, keep. Else, revert swap AND any reductions?
                        # Reverting reductions is hard.
                        # Alternative: Don't reduce inside the swap loop. 
                        # Just do swaps, then reduce at the end.
                        # But swaps might create redundancies that only become visible after reduction.
                        
                        # Let's try a different approach for swaps:
                        # Only swap if it doesn't increase size (it doesn't, 1-for-1).
                        # Then, after a batch of swaps, run reduction.
                        # If size decreases, keep. Else, revert all swaps?
                        # This is complex.
                        
                        # Let's stick to: Swap, Reduce, If better keep, else Revert Swap and Revert Reduction?
                        # We can't easily revert reduction.
                        
                        # New Plan:
                        # 1. Save state.
                        # 2. Swap.
                        # 3. Reduce.
                        # 4. If size < saved_size: Keep.
                        # 5. Else: Restore state.
                        
                        # But we already modified cover_count and lists.
                        # Let's implement a "try_swap" that returns success/fail and modifies state only on success.
                        # This is safer.
                        pass # We'll implement a safer version below if this is too slow/buggy.
                        # For now, assume the revert logic above is sufficient if we don't reduce inside.
                        # Let's remove the reduce_solution() call inside the swap loop for simplicity and safety.
                        # We will reduce after the swap loop.
                        
                        # Actually, the code above did call reduce_solution().
                        # Let's restart the local search logic in a safer way.
                        break # Break inner loop to restart logic? No, let's just break and rely on outer loop.
                        
            if not improved and time_left() < 10:
                break
                
        # Final reduction
        reduce_solution()
        
        return current_blocks

    # Better Local Search Implementation
    def local_search_v2(initial_blocks):
        if time_left() < 5:
            return initial_blocks
            
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Compute cover counts
        cover_count = [0] * n_t_subs
        for b in current_blocks:
            for t_idx in block_t_subs[b]:
                cover_count[t_idx] += 1
                
        def try_reduce():
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in current_blocks:
                    is_redundant = True
                    for t_idx in block_t_subs[b]:
                        if cover_count[t_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for t_idx in block_t_subs[b]:
                            cover_count[t_idx] -= 1
                        changed = True
                    else:
                        new_blocks.append(b)
                if changed:
                    current_blocks[:] = new_blocks
                    current_set.clear()
                    current_set.update(new_blocks)
                    
        try_reduce()
        
        # 1-Swap Loop
        # We try to perform swaps that might lead to better reductions.
        # Since we can't easily revert reductions, we will:
        # 1. Take a snapshot of the solution (list of blocks).
        # 2. Perform a swap.
        # 3. Reduce.
        # 4. If size is smaller, accept.
        # 5. Else, restore snapshot.
        
        max_iterations = 100
        for _ in range(max_iterations):
            if time_left() < 10:
                break
                
            snapshot_blocks = list(current_blocks)
            snapshot_set = set(current_blocks)
            snapshot_cover = list(cover_count)
            
            # Pick a random block to remove
            if not current_blocks:
                break
            b_out = random.choice(current_blocks)
            
            critical_ts = [t_idx for t_idx in block_t_subs[b_out] if cover_count[t_idx] == 1]
            if not critical_ts:
                continue
                
            # Find candidates
            candidates = set(t_sub_blocks[critical_ts[0]]) - snapshot_set
            for t_idx in critical_ts[1:]:
                candidates &= set(t_sub_blocks[t_idx])
            if not candidates:
                continue
                
            b_in = random.choice(list(candidates))
            
            # Apply Swap
            # Remove b_out
            for t_idx in block_t_subs[b_out]:
                cover_count[t_idx] -= 1
            current_blocks.remove(b_out)
            current_set.remove(b_out)
            
            # Add b_in
            for t_idx in block_t_subs[b_in]:
                cover_count[t_idx] += 1
            current_blocks.append(b_in)
            current_set.add(b_in)
            
            # Reduce
            try_reduce()
            
            if len(current_blocks) < len(snapshot_blocks):
                # Accept
                continue
            else:
                # Revert
                current_blocks[:] = snapshot_blocks
                current_set = snapshot_set
                cover_count[:] = snapshot_cover
                # If no improvement, we might want to perturb or break?
                # For now, just continue to next iteration.
                # To avoid infinite loop of no-improvement, we could break after N no-improvements.
                # But randomization in b_in and b_out should help.
                pass
                
        try_reduce()
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    while time_left() > 20:
        sol = greedy_construct()
        if sol is None:
            break
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
            # Local Search
            if time_left() > 30:
                improved_sol = local_search_v2(sol)
                if len(improved_sol) < best_len:
                    best_len = len(improved_sol)
                    best_solution = improved_sol
                    
        # If we have a very good solution, we might want to spend more time?
        # For now, just keep searching.
        
    if best_solution is None:
        # Fallback
        sol = greedy_construct()
        if sol:
            best_solution = sol
        else:
            return []

    return [k_subs[i] for i in best_solution]

if __name__ == "__main__":
    # Test with small parameters to ensure it runs
    # construct(19, 8, 5)
    pass
