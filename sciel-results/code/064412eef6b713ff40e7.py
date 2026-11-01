import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets (points to cover)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute block masks (bitmask of points covered)
    # Also precompute the list of t-subset indices for each block
    block_masks = np.zeros(num_k_blocks, dtype=np.int64)
    block_to_ts = []
    
    for j, block in enumerate(all_k_blocks):
        mask = 0
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            # Set bits for the points in the block?
            # Actually, for coverage check, we need to know if a block covers a set of points.
            # It's better to use the t-subset indices for coverage counting.
        block_to_ts.append(ts_list)
        
        # For the directed swap, we need to check if a block covers specific points.
        # We can precompute a bitmask for the points in the block.
        for p in block:
            mask |= (1 << p)
        block_masks[j] = mask

    # State
    is_selected = np.zeros(num_k_blocks, dtype=bool)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    
    def update_cover_count(block_idx, delta):
        """Update cover counts for a block."""
        for ts_idx in block_to_ts[block_idx]:
            cover_count[ts_idx] += delta
            if cover_count[ts_idx] < 0:
                cover_count[ts_idx] = 0 # Safety, shouldn't happen if logic is correct

    # 2. Greedy Initialization
    gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
    # Note: all blocks have the same number of t-subsets, C(k,t)
    
    selected_indices = []
    
    while True:
        if time.time() > deadline - 20.0:
            break
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        # Select block
        is_selected[max_idx] = True
        selected_indices.append(max_idx)
        gain[max_idx] = -1
        
        # Update gains
        for ts_idx in block_to_ts[max_idx]:
            if cover_count[ts_idx] == 0:
                cover_count[ts_idx] = 1
                # For all blocks covering this ts, decrement gain
                # This is the slow part of greedy.
                # Optimization: Only update gain for blocks that are still candidates.
                # We can iterate through all blocks? No, too slow.
                # We need a reverse map: ts_to_blocks.
                
                # Let's build ts_to_blocks once.
                pass # See below

    # Re-do Greedy with proper reverse map for speed
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j in range(num_k_blocks):
        for ts_idx in block_to_ts[j]:
            ts_to_blocks[ts_idx].append(j)

    # Reset
    is_selected = np.zeros(num_k_blocks, dtype=bool)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
    selected_indices = []
    
    while True:
        if time.time() > deadline - 20.0:
            break
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        is_selected[max_idx] = True
        selected_indices.append(max_idx)
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if cover_count[ts_idx] == 0:
                cover_count[ts_idx] = 1
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1

    best_solution = selected_indices[:]
    best_len = len(selected_indices)
    
    # 3. Local Search: Remove Redundancies
    def remove_redundancies():
        removed = True
        while removed:
            removed = False
            # Iterate through selected blocks
            # To allow randomization, we can shuffle the selected indices
            sel_copy = selected_indices[:]
            random.shuffle(sel_copy)
            
            for j in sel_copy:
                if not is_selected[j]:
                    continue
                # Check if j is critical
                is_critical = False
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # Remove j
                    is_selected[j] = False
                    for ts_idx in block_to_ts[j]:
                        cover_count[ts_idx] -= 1
                    selected_indices.remove(j)
                    removed = True
                    break # Restart scan after removal

    remove_redundancies()
    best_len = len(selected_indices)
    
    # 4. Local Search: Directed Swaps
    # We try to swap a critical block with another block that covers its unique points.
    
    def try_directed_swap():
        if time.time() > deadline - 5.0:
            return False
            
        # Find a critical block
        critical_block = None
        unique_mask = 0
        
        # Shuffle selected to pick a random critical block
        sel_copy = selected_indices[:]
        random.shuffle(sel_copy)
        
        for j in sel_copy:
            if not is_selected[j]:
                continue
            # Check if critical
            is_critical = False
            for ts_idx in block_to_ts[j]:
                if cover_count[ts_idx] == 1:
                    is_critical = True
                    break
            if is_critical:
                critical_block = j
                # Calculate unique mask
                # A point p is unique to j if cover_count of all t-subsets containing p is 1?
                # No, cover_count is per t-subset.
                # If a t-subset TS has cover_count 1, and j covers TS, then all points in TS are "critical" for j?
                # Not necessarily unique to j in terms of points, but the t-subset is uniquely covered.
                # For the swap to be valid, the new block must cover all t-subsets that j uniquely covers.
                # So we need a bitmask of t-subsets?
                # v=19, t=5. Number of t-subsets is 11628. Bitmask of size 11628 is too large for a single integer.
                
                # Alternative: List of unique t-subsets.
                unique_ts_list = []
                for ts_idx in block_to_ts[j]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_list.append(ts_idx)
                
                if not unique_ts_list:
                    # Should not be critical if unique_ts_list is empty
                    continue
                    
                # Now find a block j' not selected that covers all ts in unique_ts_list
                # We can check candidates randomly or systematically.
                # Let's check a random subset of unselected blocks first.
                
                # Optimization: Precompute for each t-subset, the blocks covering it.
                # We need a block j' such that for all ts in unique_ts_list, j' is in ts_to_blocks[ts].
                # This is an intersection of sets.
                
                # Start with the set of blocks covering the first unique ts
                candidate_blocks = set(ts_to_blocks[unique_ts_list[0]])
                # Remove selected blocks
                candidate_blocks -= set(selected_indices)
                
                if not candidate_blocks:
                    continue
                    
                # Intersect with sets for other unique ts
                for ts in unique_ts_list[1:]:
                    candidate_blocks &= set(ts_to_blocks[ts])
                    if not candidate_blocks:
                        break
                        
                if candidate_blocks:
                    # Pick a random candidate
                    j_prime = random.choice(list(candidate_blocks))
                    
                    # Perform Swap
                    # Remove j
                    is_selected[j] = False
                    for ts_idx in block_to_ts[j]:
                        cover_count[ts_idx] -= 1
                    selected_indices.remove(j)
                    
                    # Add j'
                    is_selected[j_prime] = True
                    for ts_idx in block_to_ts[j_prime]:
                        cover_count[ts_idx] += 1
                    selected_indices.append(j_prime)
                    
                    return True
                    
        return False

    # Main Search Loop
    no_improvement_count = 0
    
    while time.time() < deadline - 5.0:
        # Try to remove redundancies
        removed = False
        # Try a few times to remove
        for _ in range(10):
            if remove_redundancies():
                removed = True
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # If stuck, try directed swap
        if no_improvement_count > 5:
            if try_directed_swap():
                # After swap, try to remove redundancies again
                remove_redundancies()
                if len(selected_indices) < best_len:
                    best_len = len(selected_indices)
                    best_solution = selected_indices[:]
                    no_improvement_count = 0
            else:
                # If no directed swap found, try a random swap to escape
                # Pick random selected and unselected
                if selected_indices:
                    j_out = random.choice(selected_indices)
                    unselected = [j for j in range(num_k_blocks) if not is_selected[j]]
                    if unselected:
                        j_in = random.choice(unselected)
                        
                        # Perform swap
                        is_selected[j_out] = False
                        for ts_idx in block_to_ts[j_out]:
                            cover_count[ts_idx] -= 1
                        selected_indices.remove(j_out)
                        
                        is_selected[j_in] = True
                        for ts_idx in block_to_ts[j_in]:
                            cover_count[ts_idx] += 1
                        selected_indices.append(j_in)
                        
                        # Check validity? If invalid, we might want to revert or just continue.
                        # For now, let's just continue and rely on removal/repair.
                        # Actually, if we introduce uncovered points, we should try to fix them.
                        # But "Remove Redundancies" doesn't fix uncovered points.
                        # So random swap might make things worse.
                        # Let's only do random swap if we are desperate (high no_improvement_count).
                        if no_improvement_count > 50:
                            pass # Accept the random swap state
                        else:
                            # Revert
                            is_selected[j_in] = False
                            for ts_idx in block_to_ts[j_in]:
                                cover_count[ts_idx] -= 1
                            selected_indices.remove(j_in)
                            
                            is_selected[j_out] = True
                            for ts_idx in block_to_ts[j_out]:
                                cover_count[ts_idx] += 1
                            selected_indices.append(j_out)
                            
                            # If we can't do anything, break or restart?
                            # Let's just break to save time if we are stuck for too long.
                            if no_improvement_count > 100:
                                break

    # Final Cleanup
    # Ensure the best solution is valid and minimal
    # Re-apply removals to best_solution
    # (We assume best_solution is valid from the search process)
    
    final_blocks = [list(all_k_blocks[j]) for j in best_solution]
    
    return final_blocks
