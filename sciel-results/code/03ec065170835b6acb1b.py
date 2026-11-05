import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate all possible k-blocks and map them to their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each t-subset to an index 0..C(v,t)-1
    # This allows us to use a flat array for coverage tracking
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each block, the indices of the t-subsets it covers
    # Shape: (num_blocks, C(k,t))
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_subset_indices = np.empty((num_blocks, k_choose_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_subset_indices[i, j] = t_subset_to_idx[subset]
            
    # 2. Greedy Construction
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for all blocks: how many uncovered subsets does each block cover?
        # uncovered[indices] gathers the boolean values, .sum(axis=1) counts True values
        gains = uncovered[block_subset_indices].sum(axis=1)
        
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain == 0:
            # Should not happen if uncovered.any() is true, unless there's a bug
            # or we are in a state where no block covers any new subset (impossible for valid covering)
            break
            
        selected_blocks.append(max_gain_idx)
        
        # Update uncovered status
        uncovered[block_subset_indices[max_gain_idx]] = False
        
    # 3. Redundancy Removal
    # We maintain a count of how many selected blocks cover each t-subset
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_count[block_subset_indices[b_idx]] += 1
        
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            # Check if this block is redundant
            # A block is redundant if all its subsets are covered by at least one OTHER block
            # i.e., cover_count[subset] > 1 for all subsets in this block
            # We temporarily assume this block is removed to check? No, just check if min count > 1.
            # If min count > 1, then even if we remove this block, counts remain >= 1.
            
            # Get indices of subsets covered by this block
            sub_indices = block_subset_indices[b_idx]
            
            # Check if any subset is covered ONLY by this block
            # If cover_count[subset] == 1, this block is essential for that subset.
            # If min(cover_count[sub_indices]) > 1, the block is redundant.
            
            if np.min(cover_count[sub_indices]) > 1:
                # Redundant: decrement counts and skip adding to new list
                cover_count[sub_indices] -= 1
                changed = True
            else:
                # Essential: keep it
                new_selected.append(b_idx)
                
        selected_blocks = new_selected
    
    # 4. Local Search: Try to swap blocks to reduce total count
    # After greedy + redundancy removal, try 1-for-1 swaps and then re-check redundancy
    selected_set = set(selected_blocks)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each block and see if coverage is still complete
        # Then try to add back a different block that might allow further removals
        for i in range(len(selected_blocks)):
            if time.time() > deadline:
                break
            
            b_idx = selected_blocks[i]
            sub_indices = block_subset_indices[b_idx]
            
            # Check if this block is essential (any subset only covered by it)
            if np.min(cover_count[sub_indices]) <= 1:
                continue
            
            # Try removing this block
            cover_count[sub_indices] -= 1
            new_selected = selected_blocks[:i] + selected_blocks[i+1:]
            
            # Check if coverage is still complete
            # A subset is uncovered if cover_count[subset] == 0
            if np.any(cover_count == 0):
                # Not valid, put it back
                cover_count[sub_indices] += 1
                continue
            
            # Valid removal! Try to see if we can now remove more blocks
            # Re-run redundancy removal on the new set
            changed2 = True
            while changed2 and time.time() < deadline:
                changed2 = False
                new_new_selected = []
                for j, b_idx2 in enumerate(new_selected):
                    sub_indices2 = block_subset_indices[b_idx2]
                    if np.min(cover_count[sub_indices2]) > 1:
                        cover_count[sub_indices2] -= 1
                        changed2 = True
                    else:
                        new_new_selected.append(b_idx2)
                
                new_selected = new_new_selected
            
            if len(new_selected) < len(selected_blocks):
                selected_blocks = new_selected
                selected_set = set(selected_blocks)
                improved = True
                break
    
    # Final redundancy removal pass
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_count[block_subset_indices[b_idx]] += 1
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            sub_indices = block_subset_indices[b_idx]
            if np.min(cover_count[sub_indices]) > 1:
                cover_count[sub_indices] -= 1
                changed = True
            else:
                new_selected.append(b_idx)
        selected_blocks = new_selected
    
    # 5. Convert indices back to lists of points
    result_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    return result_blocks
