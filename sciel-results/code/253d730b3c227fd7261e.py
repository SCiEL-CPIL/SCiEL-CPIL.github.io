import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute masks and t-subset indices for each block
    block_masks = [0] * num_k_subsets
    block_t_indices = [[] for _ in range(num_k_subsets)]
    
    for i, block in enumerate(all_k_subsets):
        mask = 0
        indices = []
        for sub in itertools.combinations(block, t):
            idx = t_to_idx[sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
        
    # Greedy Construction
    uncovered = (1 << num_t_subsets) - 1
    selected_blocks = []
    selected_set = set()
    cover_count = [0] * num_t_subsets
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(num_k_subsets):
            if i in selected_set:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_blocks.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_masks[best_idx]
        
        for idx in block_t_indices[best_idx]:
            cover_count[idx] += 1
            
    # Local Search: Reduction and Swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Reduction
        i = 0
        while i < len(selected_blocks):
            b_idx = selected_blocks[i]
            # Check if redundant
            redundant = True
            for idx in block_t_indices[b_idx]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove b_idx
                selected_blocks.pop(i)
                selected_set.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    cover_count[idx] -= 1
                improved = True
                # Do not increment i, check next element at same position
            else:
                i += 1
        
        if time.time() >= deadline:
            break
            
        # Pass 2: Swap
        # Try to swap blocks to potentially enable further reductions
        # We iterate through blocks and try to find a replacement
        # Note: A swap might not immediately reduce size, but can change the structure
        # to allow a redundant block to appear.
        
        swapped = False
        for i in range(len(selected_blocks)):
            if time.time() >= deadline:
                break
                
            b_idx = selected_blocks[i]
            
            # Calculate deficit
            deficit_mask = 0
            for idx in block_t_indices[b_idx]:
                if cover_count[idx] == 1:
                    deficit_mask |= (1 << idx)
            
            if deficit_mask == 0:
                # Redundant, should have been caught, but handle
                selected_blocks.pop(i)
                selected_set.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    cover_count[idx] -= 1
                improved = True
                swapped = True
                break # Restart loop
            
            # Try to find replacement j not in selected_set
            # such that deficit_mask is covered by block j
            # i.e., deficit_mask & ~block_masks[j] == 0
            for j in range(num_k_subsets):
                if j in selected_set:
                    continue
                if (deficit_mask & ~block_masks[j]) == 0:
                    # Swap b_idx with j
                    selected_blocks[i] = j
                    selected_set.remove(b_idx)
                    selected_set.add(j)
                    
                    # Update cover_count
                    for idx in block_t_indices[b_idx]:
                        cover_count[idx] -= 1
                    for idx in block_t_indices[j]:
                        cover_count[idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart outer loop to re-evaluate reductions
            
            if swapped:
                break # Break inner loop and restart outer for reduction pass
        
        # If no swap happened, we might be stuck in a local minimum for 1-swap.
        # We could add 2-swap or random restarts, but 1-swap + reduction is a good baseline.
        
    # Convert indices back to blocks
    result = [list(all_k_subsets[i]) for i in selected_blocks]
    return result
