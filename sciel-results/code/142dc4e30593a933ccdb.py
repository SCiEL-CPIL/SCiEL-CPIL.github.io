import itertools
import time
import random

def construct(v, k, t):
    # Parameters for C(19, 8, 5)
    # v=19, k=8, t=5
    
    # Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(t_subsets)
    sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(k_blocks)
    
    # Compute bitmask for each k-block
    # Bit i is set if t-subset i is contained in the k-block
    block_masks = [0] * num_k_blocks
    for idx, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << sub_to_idx[t_sub])
        block_masks[idx] = mask
        
    # Greedy Initialization
    uncovered = (1 << num_t_subsets) - 1
    current_indices = []
    current_masks = []
    
    # To speed up greedy, we can just scan all blocks each time
    # 75k blocks * ~300 steps = 22.5M ops. Fast enough.
    while uncovered:
        best_idx = -1
        best_cover = 0
        for i in range(num_k_blocks):
            cover = (block_masks[i] & uncovered).bit_count()
            if cover > best_cover:
                best_cover = cover
                best_idx = i
        if best_idx == -1:
            break # Should not happen if problem is solvable
        current_indices.append(best_idx)
        current_masks.append(block_masks[best_idx])
        uncovered &= ~block_masks[best_idx]
        
    # Local Search
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Deletion Phase
        # Check if any block is redundant
        new_indices = []
        new_masks = []
        for i in range(len(current_indices)):
            m = current_masks[i]
            others = 0
            for j in range(len(current_indices)):
                if i != j:
                    others |= current_masks[j]
            if (m & ~others) == 0:
                improved = True
            else:
                new_indices.append(current_indices[i])
                new_masks.append(current_masks[i])
                
        if improved:
            current_indices = new_indices
            current_masks = new_masks
            continue # Restart loop to check for more deletions or swaps
            
        # Swap Phase
        # Try to replace a block with another that covers its unique subsets
        swapped = False
        for i in range(len(current_indices)):
            m = current_masks[i]
            others = 0
            for j in range(len(current_indices)):
                if i != j:
                    others |= current_masks[j]
            unique = m & ~others
            
            if unique == 0:
                # Should have been deleted, but handle just in case
                new_indices = [current_indices[j] for j in range(len(current_indices)) if j != i]
                new_masks = [current_masks[j] for j in range(len(current_indices)) if j != i]
                current_indices = new_indices
                current_masks = new_masks
                swapped = True
                break
            
            # Find a candidate block c not in current set such that unique is subset of block_masks[c]
            # We iterate through all blocks.
            # Optimization: We can skip blocks that are already in current set.
            current_set = set(current_indices)
            
            best_candidate = -1
            # Heuristic: Prefer candidate with max total coverage? Or just first found?
            # Let's try to find one that maximizes (block_masks[c] & ~others).bit_count()
            # This might lead to better future deletions.
            max_potential = 0
            for c in range(num_k_blocks):
                if c in current_set:
                    continue
                if (unique & ~block_masks[c]) != 0:
                    continue
                # This candidate covers all unique subsets of block i
                # Calculate potential gain: how many NEW subsets does it cover compared to 'others'?
                # Actually, it covers 'unique' and some subset of 'others'.
                # The new 'others' would be 'others' | block_masks[c].
                # The goal is to enable deletions.
                # A good proxy is to pick the candidate that covers the most subsets in 'others' that are currently covered by multiple blocks?
                # Or simply pick the one with the highest total popcount.
                # Let's just pick the first one for simplicity, or randomize?
                # Randomization might help escape local minima.
                # But deterministic is safer for reproducibility.
                # Let's pick the one with max popcount(block_masks[c]) to keep blocks "strong".
                pop = block_masks[c].bit_count()
                if pop > max_potential:
                    max_potential = pop
                    best_candidate = c
            
            if best_candidate != -1:
                current_indices[i] = best_candidate
                current_masks[i] = block_masks[best_candidate]
                swapped = True
                break
        
        if not swapped:
            break # No improvement possible
            
    # Construct final blocks
    final_blocks = [k_blocks[i] for i in current_indices]
    
    return final_blocks
