import itertools
import numpy as np

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset and list of t-subset indices
    masks = [0] * n_k_subsets
    block_t_subsets = [[] for _ in range(n_k_subsets)]
    for i, block in enumerate(k_subsets):
        mask = 0
        idx_list = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            idx_list.append(idx)
        masks[i] = mask
        block_t_subsets[i] = idx_list
        
    # Precompute for each t-subset index, which blocks contain it
    # This helps in finding replacements for essential subsets
    # t_to_blocks[idx] = list of block indices
    t_to_blocks = [[] for _ in range(n_t_subsets)]
    for i in range(n_k_subsets):
        for idx in block_t_subsets[i]:
            t_to_blocks[idx].append(i)

    # 4. Greedy Set Cover
    uncovered = (1 << n_t_subsets) - 1
    selected_indices = []
    used_blocks = [False] * n_k_subsets
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        
        for i in range(n_k_subsets):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    # Initialize coverage counts
    coverage_count = [0] * n_t_subsets
    for i in selected_indices:
        for idx in block_t_subsets[i]:
            coverage_count[idx] += 1
            
    # 5. Local Search: Redundancy Removal and Swapping
    changed = True
    while changed:
        changed = False
        # Iterate through a copy of selected_indices to avoid mutation issues during iteration
        # We break and restart on any change
        for pos, block_idx in enumerate(list(selected_indices)):
            # Find essential subsets for this block
            essential = [idx for idx in block_t_subsets[block_idx] if coverage_count[idx] == 1]
            
            if not essential:
                # Redundant: Remove it
                selected_indices.pop(pos)
                used_blocks[block_idx] = False
                for idx in block_t_subsets[block_idx]:
                    coverage_count[idx] -= 1
                changed = True
                break # Restart loop
            
            else:
                # Try to find a replacement block that covers all essential subsets
                # Intersection of t_to_blocks for all essential subsets
                # Start with the block list of the first essential subset
                candidates = set(t_to_blocks[essential[0]])
                
                for e_idx in essential[1:]:
                    if not candidates:
                        break
                    candidates &= set(t_to_blocks[e_idx])
                
                if candidates:
                    # Filter candidates: must not be the current block, and must not be already selected
                    # Actually, if it's already selected, swapping is a no-op or complex. 
                    # We only want to swap with an UNSELECTED block.
                    valid_candidates = [c for c in candidates if not used_blocks[c] and c != block_idx]
                    
                    if valid_candidates:
                        # Pick one (e.g., the one with max total coverage or just the first)
                        # To be safe, pick the one that covers the most NEW stuff? 
                        # Or just the first one.
                        new_block_idx = valid_candidates[0]
                        
                        # Perform Swap
                        # Remove old
                        for idx in block_t_subsets[block_idx]:
                            coverage_count[idx] -= 1
                        used_blocks[block_idx] = False
                        
                        # Add new
                        for idx in block_t_subsets[new_block_idx]:
                            coverage_count[idx] += 1
                        used_blocks[new_block_idx] = True
                        
                        selected_indices[pos] = new_block_idx
                        changed = True
                        break # Restart loop

    # 6. Construct final list of blocks
    return [list(k_subsets[i]) for i in selected_indices]
