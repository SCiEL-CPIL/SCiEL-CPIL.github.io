import itertools
import time
import random

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
    
    # Precompute (t-1) subsets for each possible partial block size? 
    # Actually, let's precompute masks for all (t-1)-subsets of all (t-1)-combinations of range(v)
    # But we need them for the current block only.
    
    # Helper to get list of t-subset masks for a block
    def get_block_t_masks(block):
        masks = []
        for comb in itertools.combinations(block, t):
            m = 0
            for p in comb:
                m |= (1 << p)
            masks.append(m)
        return masks

    best_blocks = None
    best_len = float('inf')
    
    # We use a set of uncovered indices for the greedy phase
    # Using indices instead of masks for faster set operations? 
    # Actually, masks are integers, so they are hashable and fast.
    # But we can use a boolean array for O(1) access and faster iteration if needed.
    # Let's stick to set of masks for simplicity, but use mask_to_idx for counting if needed.
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        uncovered = set(all_t_subs)
        blocks = []
        
        # Greedy Phase
        while uncovered and time.time() < deadline:
            # Pick an uncovered subset
            # min() on a set of ints is fast.
            first_mask = min(uncovered)
            
            # Extract points from mask
            init_points = []
            m = first_mask
            idx = 0
            while m:
                if m & 1:
                    init_points.append(idx)
                m >>= 1
                idx += 1
            
            block = init_points
            block_set = set(block)
            
            # Grow the block
            while len(block) < k and time.time() < deadline:
                best_x = -1
                best_gain = -1
                best_ties = []
                
                # Generate all (t-1) subsets of current block
                t_minus_1_subs = []
                for comb in itertools.combinations(block, t - 1):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    t_minus_1_subs.append(mask)
                
                # Evaluate gain for each candidate x
                # Optimization: Precompute the list of (sub_mask | x_bit) for each x?
                # No, just loop.
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    x_bit = 1 << x
                    for sub_mask in t_minus_1_subs:
                        if (sub_mask | x_bit) in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_ties = [x]
                    elif gain == best_gain:
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
            
            # Update uncovered
            for comb in itertools.combinations(block, t):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                uncovered.discard(mask)
        
        if not uncovered:
            # Valid covering found
            # Redundancy Removal
            # Count coverage for each t-subset using indices for speed
            cover_count = [0] * num_t_subs
            
            for b in blocks:
                for comb in itertools.combinations(b, t):
                    mask = 0
                    for p in comb:
                        mask |= (1 << p)
                    cover_count[mask_to_idx[mask]] += 1
            
            # Try to remove redundant blocks
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    b = blocks[i]
                    redundant = True
                    affected_indices = []
                    for comb in itertools.combinations(b, t):
                        mask = 0
                        for p in comb:
                            mask |= (1 << p)
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
        # Fallback: return empty list or a trivial covering if possible?
        # If greedy didn't finish, we might have an incomplete covering.
        # But the task requires a valid covering.
        # If we didn't find one, we return [] which is invalid (score 0).
        # However, with v=19, k=8, t=5, greedy usually finds a covering.
        return []
        
    return best_blocks
