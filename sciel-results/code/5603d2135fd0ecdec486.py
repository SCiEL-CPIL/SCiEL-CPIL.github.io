import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets as bitmasks and map to indices
    all_t_subs = []
    mask_to_idx = {}
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subs.append(mask)
        mask_to_idx[mask] = len(all_t_subs) - 1
    
    num_t_subs = len(all_t_subs)
    max_mask = 1 << v
    
    # Precompute (t-1) subset masks for each point combination
    # For a given block, we need masks of all (t-1) subsets
    # We'll compute these on the fly but cache per block
    
    best_blocks = None
    best_len = float('inf')
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Use a boolean array for O(1) lookup
        is_uncovered = np.ones(num_t_subs, dtype=bool)
        
        blocks = []
        
        # Greedy Phase
        while time.time() < deadline:
            # Pick an uncovered subset - find first True in is_uncovered
            uncovered_indices = np.where(is_uncovered)[0]
            if len(uncovered_indices) == 0:
                break  # All covered
            
            first_idx = uncovered_indices[0]
            first_mask = all_t_subs[first_idx]
            
            # Extract points from mask
            init_points = []
            m = first_mask
            idx = 0
            while m:
                if m & 1:
                    init_points.append(idx)
                m >>= 1
                idx += 1
            
            block = list(init_points)
            block_set = set(block)
            
            # Precompute masks for all (t-1) subsets of current block
            def get_t_minus_1_masks(blk):
                masks = []
                for comb in itertools.combinations(blk, t - 1):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    masks.append(mask)
                return masks
            
            # Grow the block
            while len(block) < k and time.time() < deadline:
                t_minus_1_subs = get_t_minus_1_masks(block)
                t_minus_1_arr = np.array(t_minus_1_subs, dtype=np.uint64)
                
                # Evaluate gain for each candidate x using vectorized operations
                best_x = -1
                best_gain = -1
                best_ties = []
                
                # For each candidate x not in block, compute gain
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    x_bit = np.uint64(1 << x)
                    # Combine each (t-1) subset mask with x_bit
                    combined = t_minus_1_arr | x_bit
                    # Check which are uncovered
                    gains = is_uncovered[combined].sum()
                    
                    if gains > best_gain:
                        best_gain = gains
                        best_x = x
                        best_ties = [x]
                    elif gains == best_gain:
                        best_ties.append(x)
                
                if best_x == -1:
                    # No gain, just fill remaining
                    for x in range(v):
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    break
                
                # Random tie-breaking
                if len(best_ties) > 1:
                    best_x = random.choice(best_ties)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update is_uncovered - vectorized
            block_t_masks = []
            for comb in itertools.combinations(block, t):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                block_t_masks.append(mask_to_idx[mask])
            if block_t_masks:
                is_uncovered[np.array(block_t_masks, dtype=np.int64)] = False
        
        # Check if valid
        valid = not is_uncovered.any()
        
        if valid:
            # Redundancy Removal
            # Build coverage count array
            cover_count = np.zeros(num_t_subs, dtype=np.int32)
            
            # Precompute t-masks for each block
            block_t_mask_lists = []
            for b in blocks:
                masks = []
                for comb in itertools.combinations(b, t):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    masks.append(mask_to_idx[mask])
                block_t_mask_lists.append(masks)
                cover_count[np.array(masks, dtype=np.int64)] += 1
            
            # Try to remove redundant blocks - more aggressive
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Try to remove blocks in order of lowest coverage contribution
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    affected_indices = block_t_mask_lists[i]
                    if not affected_indices:
                        continue
                    
                    # Check if any subset is only covered by this block
                    arr = np.array(affected_indices, dtype=np.int64)
                    if (cover_count[arr] <= 1).any():
                        continue  # Not redundant
                    
                    # This block is redundant - remove it
                    cover_count[arr] -= 1
                    blocks.pop(i)
                    block_t_mask_lists.pop(i)
                    changed = True
                    break  # Restart scan since indices shifted
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
            if best_len <= 210:
                break

        # Check time for next attempt
        if time.time() > deadline - 2.0:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
