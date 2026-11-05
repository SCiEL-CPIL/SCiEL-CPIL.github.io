import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Map each block to indices of t-subsets it covers
    block_to_subsets = [None] * num_blocks
    for i, block in enumerate(all_blocks):
        subs = [t_subset_to_idx[subset] for subset in itertools.combinations(block, t)]
        block_to_subsets[i] = np.array(subs, dtype=np.int32)
        
    # Map each t-subset to indices of blocks that cover it
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for s_idx in block_to_subsets[i]:
            subset_to_blocks[s_idx].append(i)

    # 2. Greedy Construction
    uncovered = np.ones(num_t_subsets, dtype=bool)
    gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    selected_blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain == 0:
            break
            
        selected_blocks.append(max_gain_idx)
        
        b_subs = block_to_subsets[max_gain_idx]
        uncovered[b_subs] = False
        
        # Update gains
        for s_idx in b_subs:
            for b_prime in subset_to_blocks[s_idx]:
                if b_prime != max_gain_idx:
                    gains[b_prime] -= 1

    # 3. Local Search
    coverage_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_blocks:
        coverage_counts[block_to_subsets[b_idx]] += 1
        
    selected_set = set(selected_blocks)
    
    def is_redundant(b_idx):
        subs = block_to_subsets[b_idx]
        return np.min(coverage_counts[subs]) > 1

    def deletion_pass():
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            random.shuffle(selected_blocks)
            for i in range(len(selected_blocks)):
                b_idx = selected_blocks[i]
                if is_redundant(b_idx):
                    coverage_counts[block_to_subsets[b_idx]] -= 1
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    break

    def find_swap_candidate(old_idx):
        subs = block_to_subsets[old_idx]
        criticals = [s for s in subs if coverage_counts[s] == 1]
        
        if not criticals:
            return None
            
        candidates = None
        for s in criticals:
            if candidates is None:
                candidates = set(subset_to_blocks[s])
            else:
                candidates.intersection_update(subset_to_blocks[s])
                if not candidates:
                    return None
        
        candidates.discard(old_idx)
        candidates -= selected_set
        
        if not candidates:
            return None
            
        return random.choice(list(candidates))

    # Main Loop
    while time.time() < deadline - 2:
        # Deletion
        deletion_pass()
        
        # Swap
        # We iterate over a snapshot to avoid issues if list changes
        # But deletion_pass modifies selected_blocks.
        # So we take a snapshot of indices.
        current_len = len(selected_blocks)
        for i in range(current_len):
            if time.time() > deadline - 2:
                break
                
            # Get the block at index i. Note: indices might shift if deletion happened in previous swap?
            # No, deletion_pass is called at start of loop.
            # Inside the swap loop, we don't call deletion_pass until AFTER a swap.
            # So indices are stable during the swap loop iteration?
            # If we swap, we replace element at i. We do NOT delete in the swap step itself.
            # We only delete in deletion_pass.
            # So iterating 0 to current_len-1 is safe as long as we don't change length.
            # Swap changes content, not length.
            
            old_idx = selected_blocks[i]
            new_idx = find_swap_candidate(old_idx)
            
            if new_idx is not None:
                # Perform Swap
                coverage_counts[block_to_subsets[old_idx]] -= 1
                coverage_counts[block_to_subsets[new_idx]] += 1
                
                selected_set.remove(old_idx)
                selected_set.add(new_idx)
                selected_blocks[i] = new_idx
                
                # After swap, try to delete
                deletion_pass()
                # If deletion happened, selected_blocks length changed.
                # We break out of the swap loop to restart with new length?
                # Yes, safer to restart the inner loop.
                break 

    # Final Cleanup
    deletion_pass()
    
    result_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    return result_blocks
