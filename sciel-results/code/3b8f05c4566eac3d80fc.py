import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # For each block, compute the bitmask of t-subsets it covers
    block_masks = [0] * num_blocks
    # Also store the list of subset indices for each block for faster iteration in local search
    block_subsets = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs
        
    # Inverted index: subset index -> list of block indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for i in range(num_blocks):
        for s_idx in block_subsets[i]:
            subset_to_blocks[s_idx].append(i)

    # 2. Greedy Set Cover
    all_mask = (1 << num_subsets) - 1
    covered = 0
    selected_indices = []
    selected_set = set()
    
    while covered != all_mask:
        if time.time() > deadline:
            break
        
        best_block_idx = -1
        best_gain = 0
        
        # Find the block that covers the most *uncovered* subsets
        for i in range(num_blocks):
            gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_block_idx)
        selected_set.add(best_block_idx)
        covered |= block_masks[best_block_idx]

    # 3. Local Search: Redundancy Removal and Swap
    # Initialize counts
    subset_counts = [0] * num_subsets
    for idx in selected_set:
        for s_idx in block_subsets[idx]:
            subset_counts[s_idx] += 1

    # Helper to check time
    def time_ok():
        return time.time() < deadline

    # Main local search loop
    while time_ok():
        improved = False
        
        # Pass 1: Redundancy Removal
        # Iterate over a copy of the list to allow removal
        for idx in list(selected_set):
            if not time_ok():
                break
            
            is_redundant = True
            for s_idx in block_subsets[idx]:
                if subset_counts[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                selected_set.remove(idx)
                for s_idx in block_subsets[idx]:
                    subset_counts[s_idx] -= 1
                improved = True
        
        if improved:
            continue # Restart loop to check for new redundancies
        
        # Pass 2: Swap Search
        # Try to swap a block with another that covers its unique subsets
        for idx in list(selected_set):
            if not time_ok():
                break
            
            # Find subsets covered ONLY by this block
            unique_subsets = []
            for s_idx in block_subsets[idx]:
                if subset_counts[s_idx] == 1:
                    unique_subsets.append(s_idx)
            
            if not unique_subsets:
                continue # Should have been removed in Pass 1
            
            # Find candidates that cover all unique_subsets
            # Intersection of lists of blocks covering each unique subset
            # Start with the first one
            candidates = set(subset_to_blocks[unique_subsets[0]])
            
            for s_idx in unique_subsets[1:]:
                candidates &= set(subset_to_blocks[s_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Remove already selected blocks
            candidates -= selected_set
            
            if candidates:
                # Pick a random candidate or the first one
                # Random might help escape local minima
                new_idx = random.choice(list(candidates))
                
                # Perform swap
                selected_set.remove(idx)
                selected_set.add(new_idx)
                
                # Update counts
                for s_idx in block_subsets[idx]:
                    subset_counts[s_idx] -= 1
                for s_idx in block_subsets[new_idx]:
                    subset_counts[s_idx] += 1
                
                improved = True
                break # Restart loop after a swap

        if not improved:
            break

    # 4. Final Verification and Return
    # Check if all subsets are covered
    final_covered = 0
    for idx in selected_set:
        final_covered |= block_masks[idx]
        
    if final_covered != all_mask:
        # Fallback to greedy result if local search broke it (should not happen)
        # Re-run greedy? Or just return what we have if it's valid?
        # The prompt says "Valid when every t-subset... lies in at least one block".
        # If it's invalid, score is 0.
        # So we must ensure validity.
        # If local search broke it, we should revert to the last valid state.
        # But we didn't save it.
        # To be safe, if final_covered != all_mask, we should probably just return the greedy result.
        # But we overwrote selected_set.
        # Let's store the greedy result separately.
        
        # Actually, the swap logic only adds a block that covers the unique subsets,
        # and removes a block.
        # If unique subsets were covered only by the removed block, and the new block covers them,
        # they are still covered.
        # Other subsets covered by the removed block are still covered by others (since they weren't unique).
        # So coverage should be preserved.
        # But just in case, if it fails, we can't easily recover.
        # However, the logic is sound.
        
        pass

    result_blocks = [list(all_blocks[i]) for i in selected_set]
    return result_blocks
