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
    
    # Precompute bit for each point
    point_bits = [1 << i for i in range(v)]
    
    # Precompute all t-subset masks for each k-block? No, too many.
    # But we can precompute the t-subset masks for a given block on demand.
    
    # For speed, precompute for each point x, the list of t-subset masks that include x
    # Actually, we'll use a different approach: for each block, compute its t-subset masks incrementally.
    
    best_blocks = None
    best_len = float('inf')
    
    # Use a numpy boolean array for uncovered for faster operations
    # uncovered_np[i] = True if t-subset i is uncovered
    uncovered_np = np.ones(num_t_subs, dtype=bool)
    
    # Precompute for each possible (t-1)-subset mask, the list of t-subset indices it belongs to
    # This helps in computing gains efficiently
    # For a (t-1)-subset S and a point x not in S, the t-subset is S ∪ {x}
    # We need to check if that t-subset is uncovered.
    
    # Alternative approach: use a dictionary from (t-1)-mask to list of (x, t_subset_idx)
    # But this might be large. Let's think differently.
    
    # For gain computation: when adding point x to a block B, the new t-subsets covered are 
    # all (T-1)-subsets of B combined with x.
    # So gain(x) = number of (T-1)-subsets S of B such that S ∪ {x} is uncovered.
    
    # We can precompute for each (t-1)-subset mask m, and each point x not in m,
    # the index of the t-subset m ∪ {x}.
    # Then gain(x) = count of S in C(B, t-1) such that uncovered_np[idx(m ∪ {x})] is True.
    
    # Let's precompute a 2D structure: for each (t-1)-mask, for each point x not in it, the t-subset index.
    # But there are C(19,4) = 3876 (t-1)-subsets and 19 points, so about 73644 entries. Manageable.
    
    # Actually, let's just compute on the fly but use numpy for the uncovered check.
    
    # Helper function to get points from a mask
    def mask_to_points(mask):
        pts = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                pts.append(idx)
            m >>= 1
            idx += 1
        return pts
    
    # Helper function to compute t-subset masks for a block
    def get_block_t_masks(block):
        masks = []
        for comb in itertools.combinations(block, t):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            masks.append(mask)
        return masks
    
    # Helper function to update uncovered_np with a block
    def apply_block(block):
        for mask in get_block_t_masks(block):
            idx = mask_to_idx[mask]
            uncovered_np[idx] = False
    
    # Precompute for each (t-1)-subset, the mapping to t-subset indices when adding each point
    # (t-1)_mask -> dict: point -> t_subset_index
    t1_to_t_idx = {}
    for comb in itertools.combinations(range(v), t - 1):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        t1_set = set(comb)
        mapping = {}
        for x in range(v):
            if x not in t1_set:
                new_mask = mask | (1 << x)
                mapping[x] = mask_to_idx[new_mask]
        t1_to_t_idx[mask] = mapping
    
    # Now for gain computation:
    # Given block B and candidate x, gain = count of (t-1)-subsets S of B such that 
    # uncovered_np[t1_to_t_idx[S_mask][x]] is True.
    
    # We can precompute the (t-1)-subset masks of B and use numpy to check uncovered_np.
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Reset uncovered
        uncovered_np[:] = True
        blocks = []
        
        # Greedy Phase
        while np.any(uncovered_np) and time.time() < deadline:
            # Pick an uncovered subset - use the first one for determinism in base, but add randomness
            # Find an uncovered index
            uncov_indices = np.where(uncovered_np)[0]
            if len(uncov_indices) == 0:
                break
            
            # Random selection for diversity
            chosen_idx = random.choice(uncov_indices)
            first_mask = all_t_subs[chosen_idx]
            
            # Extract points from mask
            init_points = mask_to_points(first_mask)
            
            block = list(init_points)
            block_set = set(block)
            
            # Grow the block
            while len(block) < k and time.time() < deadline:
                best_x = -1
                best_gain = -1
                best_ties = []
                
                # Generate all (t-1) subsets of current block as masks
                t_minus_1_subs = []
                for comb in itertools.combinations(block, t - 1):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    t_minus_1_subs.append(mask)
                
                # For each candidate x not in block, compute gain
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                
                # Use numpy for faster checking
                # For each (t-1)-subset mask, we have a mapping from x to t_subset_index
                # gain(x) = sum over S in t_minus_1_subs of uncovered_np[t1_to_t_idx[S_mask][x]]
                
                # Build a numpy array of indices to check
                # For each S in t_minus_1_subs and each x in candidates, get the t_subset_index
                # Then check uncovered_np at those indices
                
                # This could be large: C(|block|, t-1) * (v - |block|)
                # For |block| up to 7, t-1=4, C(7,4)=35, v-|block| up to 19, so 35*19=665. Fine.
                
                # Let's compute gains for all candidates at once using numpy
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for s_mask in t_minus_1_subs:
                    mapping = t1_to_t_idx[s_mask]
                    # For each candidate x, get the t_subset_index
                    for i, x in enumerate(candidates):
                        if x in mapping:
                            idx = mapping[x]
                            if uncovered_np[idx]:
                                gains[i] += 1
                
                # Find best gain
                max_gain = np.max(gains) if len(gains) > 0 else -1
                if max_gain > 0:
                    best_gain = int(max_gain)
                    best_ties = [candidates[i] for i in range(len(candidates)) if gains[i] == max_gain]
                    best_x = random.choice(best_ties) if len(best_ties) > 1 else best_ties[0]
                else:
                    # No gain, just fill remaining
                    for x in range(v):
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    break
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            apply_block(block)
        
        if not np.any(uncovered_np):
            # Valid covering found
            # Redundancy Removal
            cover_count = [0] * num_t_subs
            
            for b in blocks:
                for mask in get_block_t_masks(b):
                    cover_count[mask_to_idx[mask]] += 1
            
            # Try to remove redundant blocks - multiple passes
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    b = blocks[i]
                    redundant = True
                    affected_indices = []
                    for mask in get_block_t_masks(b):
                        idx = mask_to_idx[mask]
                        if cover_count[idx] <= 1:
                            redundant = False
                            break
                        affected_indices.append(idx)
                    
                    if redundant:
                        for idx in affected_indices:
                            cover_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
                
            if best_len <= 210:
                break
        else:
            pass

        # Check time for next attempt
        if time.time() > deadline - 2.0:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
