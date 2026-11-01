import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Greedy Set Cover
    uncovered = [True] * num_t_subsets
    # gain[b] = number of uncovered t-subsets covered by block b
    gain = [0] * num_k_subsets
    
    # Initialize gain
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx]) # All are uncovered initially
        
    selected_blocks = []
    
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
            
        # Find the block with max gain
        # If tie, pick the one with smallest index for determinism
        best_block = gain.index(max_gain)
        
        selected_blocks.append(best_block)
        
        # Update gains
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                # Decrement gain for all blocks covering this t_subset
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # 3. Minimalization (Local Search: Deletion)
    # Calculate cover counts
    cover_count = [0] * num_t_subsets
    for b_idx in selected_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    # Iterative Deletion Pass
    # We repeatedly try to remove blocks until no more can be removed.
    # This handles dependencies where removing one block makes another removable.
    
    active_blocks = set(selected_blocks)
    
    # We will loop until no block is removed in a full pass
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            # Check if all t-subsets in this block are covered by >1 active block
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                changed = True

    # Convert back to list of lists
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    return result
