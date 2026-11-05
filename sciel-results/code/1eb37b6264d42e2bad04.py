import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subsets mapping (list of lists of indices)
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            block_to_subsets[b_idx].append(subset_to_idx[s])
            
    # Precompute subset to blocks mapping
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def get_greedy_solution():
        uncovered = np.ones(num_subsets, dtype=bool)
        # gains[i] is the number of currently uncovered subsets covered by block i
        gains = np.array([sum(1 for s in subs if uncovered[s]) for subs in block_to_subsets], dtype=np.int32)
        selected = []
        selected_set = set()
        
        while uncovered.any():
            # Find block with max gain
            # To avoid picking blocks already selected (gain might be stale or 0), 
            # we just pick the global max. If it's already selected, its gain should be 0 
            # if we update gains correctly. 
            # However, updating gains for all blocks is O(N_blocks * N_subsets_per_block).
            # A faster way: just pick max gain. If it's in selected_set, it means we didn't update gains properly?
            # Actually, if a block is selected, we should set its gain to 0 or ignore it.
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            if best_idx in selected_set:
                # This shouldn't happen if we set gain to 0 after selection
                gains[best_idx] = 0
                continue

            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update gains for all other blocks
            b_subs = block_to_subsets[best_idx]
            for s in b_subs:
                if uncovered[s]:
                    uncovered[s] = False
                    # Decrease gain for all blocks covering s
                    for b in subset_to_blocks[s]:
                        if b not in selected_set:
                            gains[b] -= 1
                            
        return selected

    def local_search(start_time, time_limit):
        current_set = get_greedy_solution()
        if not current_set:
            return []
            
        # Calculate initial coverage counts
        coverage_counts = np.zeros(num_subsets, dtype=np.int32)
        for b in current_set:
            for s in block_to_subsets[b]:
                coverage_counts[s] += 1
                
        current_set_list = list(current_set)
        current_set_set = set(current_set_list)
        
        best_len = len(current_set_list)
        best_set = list(current_set_list)
        
        # Helper to perform a swap
        def do_swap(old_idx, new_idx):
            old_block = current_set_list[old_idx]
            new_block = new_idx
            
            # Remove old
            for s in block_to_subsets[old_block]:
                coverage_counts[s] -= 1
            # Add new
            for s in block_to_subsets[new_block]:
                coverage_counts[s] += 1
            
            current_set_list[old_idx] = new_block
            current_set_set.discard(old_block)
            current_set_set.add(new_block)

        def do_delete(idx):
            old_block = current_set_list[idx]
            for s in block_to_subsets[old_block]:
                coverage_counts[s] -= 1
            current_set_list.pop(idx)
            current_set_set.discard(old_block)

        while time.time() - start_time < time_limit:
            # 1. Deletion Pass
            changed = True
            while changed:
                changed = False
                # Iterate in reverse to allow safe popping
                for i in range(len(current_set_list) - 1, -1, -1):
                    b = current_set_list[i]
                    redundant = True
                    for s in block_to_subsets[b]:
                        if coverage_counts[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        do_delete(i)
                        changed = True
                        # After deletion, coverage counts changed, so we must restart 
                        # to check other blocks again
                        break 
                
            if len(current_set_list) < best_len:
                best_len = len(current_set_list)
                best_set = list(current_set_list)
                
            # 2. Swap Pass
            # Try to find a swap that might allow further deletions
            # We iterate through blocks. If a block has unique subsets, we look for a replacement.
            swap_found = False
            # Shuffle indices for randomness
            indices = list(range(len(current_set_list)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                    
                old_block = current_set_list[i]
                
                # Find unique subsets for this block
                unique_subs = []
                for s in block_to_subsets[old_block]:
                    if coverage_counts[s] == 1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    # This block is redundant. The deletion pass should have caught it,
                    # but maybe it became redundant due to a previous swap in this pass?
                    # If so, delete it.
                    do_delete(i)
                    swap_found = True
                    break # Restart loop
                
                # If there are unique subsets, we need a block that covers all of them.
                # Candidates: intersection of subset_to_blocks[s] for s in unique_subs
                # Optimization: Start with the smallest list
                if not unique_subs:
                    continue
                    
                min_len = float('inf')
                min_list = None
                for s in unique_subs:
                    if len(subset_to_blocks[s]) < min_len:
                        min_len = len(subset_to_blocks[s])
                        min_list = subset_to_blocks[s]
                        
                if min_list is None:
                    continue
                    
                candidates = set(min_list)
                for s in unique_subs[1:]: # Skip the first one already used
                    candidates.intersection_update(subset_to_blocks[s])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Remove current block and any block already in the set
                candidates.discard(old_block)
                candidates.difference_update(current_set_set)
                
                if candidates:
                    # Pick a candidate. 
                    # Heuristic: Pick one that covers the most "critical" subsets? 
                    # Or just random. Let's pick random to explore.
                    new_block = random.choice(list(candidates))
                    
                    # Perform swap
                    do_swap(i, new_block)
                    swap_found = True
                    break # Restart deletion/swap loop
            
            if not swap_found and not changed:
                # No changes in this iteration, might be stuck in local minimum.
                # We can break or try a more aggressive random restart, but for time limits,
                # breaking is safer if we've converged.
                # However, "changed" is only from deletion pass. 
                # If no swap and no deletion, we are stable.
                break

        return best_set

    start_time = time.time()
    result = local_search(start_time, 115)
    
    # Convert indices to blocks
    return [list(all_blocks[i]) for i in result]
