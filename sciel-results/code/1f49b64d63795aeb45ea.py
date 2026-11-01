# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(v, k, t):
    # Deadline management: leave 10 seconds for final repair and reduction
    deadline = time.time() + 110.0
    
    # 1. Precomputation
    # Generate all t-subsets and assign each a bit position
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}
    
    # Generate all k-subsets and precompute their coverage as Python ints (bitmasks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # For each k-subset, compute the bitmask of t-subsets it covers
    # Using Python int with bit_count() for fast popcount
    block_masks = []
    for comb in all_k_subsets:
        mask = 0
        for sub in itertools.combinations(comb, t):
            mask |= (1 << tsub_to_bit[sub])
        block_masks.append(mask)
    
    # Full mask: all t-subsets uncovered
    full_mask = (1 << n_tsub) - 1
    uncovered = full_mask
    
    # State
    cover_count = [0] * n_tsub
    blocks_selected = []
    selected_set = set()
    
    # 2. Greedy Loop with Bitset Lazy Evaluation
    # For each block, gain = popcount(block_mask & uncovered)
    # We compute gains lazily and track them
    
    while uncovered and time.time() < deadline - 10.0:
        # Compute gains for all blocks
        # Using Python's int.bit_count() which is hardware-accelerated
        gains = [0] * n_ksub
        best_gain = 0
        best_idx = -1
        
        # Local variables for speed
        uncv = uncovered
        b_masks = block_masks
        
        for i in range(n_ksub):
            g = (b_masks[i] & uncv).bit_count()
            if g > best_gain:
                best_gain = g
                best_idx = i
        
        if best_gain == 0:
            break
        
        # Add the best block
        blocks_selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update cover_count and uncovered
        mask = block_masks[best_idx]
        # Extract set bits to update cover_count
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            cover_count[bit_pos] += 1
            m ^= lsb
        
        # Update uncovered
        uncovered ^= mask  # Clear bits that are now covered
        # Actually, we want to clear only the bits that were uncovered
        # uncovered &= ~mask is cleaner
        uncovered &= ~mask
    
    # 3. Repair Phase (if any t-subsets still uncovered)
    if uncovered and time.time() < deadline - 5.0:
        # Find uncovered t-subsets
        m = uncovered
        while m:
            if time.time() > deadline - 2.0:
                break
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            m ^= lsb
            
            # Get the t-subset
            tsub = tsubs[bit_pos]
            have = set(tsub)
            need = k - len(have)
            
            remaining_points = [p for p in range(v) if p not in have]
            
            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []
            
            block = sorted(list(have) + extra)
            
            # Find or create this block
            # Check if it's already in all_k_subsets
            block_tuple = tuple(block)
            if block_tuple in selected_set:
                continue
            
            # Find index in all_k_subsets
            b_idx = None
            for i, comb in enumerate(all_k_subsets):
                if comb == block_tuple:
                    b_idx = i
                    break
            
            if b_idx is not None and b_idx not in selected_set:
                blocks_selected.append(b_idx)
                selected_set.add(b_idx)
                
                # Update cover_count and uncovered
                mask = block_masks[b_idx]
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    bit_pos2 = lsb2.bit_length() - 1
                    cover_count[bit_pos2] += 1
                    m2 ^= lsb2
                
                uncovered &= ~mask
    
    # 4. Reduction Phase - Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        # Iterate in reverse to make removal easier
        for i in range(len(blocks_selected) - 1, -1, -1):
            if time.time() > deadline - 1.0:
                break
            b_idx = blocks_selected[i]
            
            # Check if redundant: any t-subset in this block has count == 1
            redundant = True
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                m ^= lsb
                if cover_count[bit_pos] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    cover_count[bit_pos] -= 1
                
                blocks_selected.pop(i)
                selected_set.remove(b_idx)
                changed = True
                break  # Restart loop
    
    # 5. Final Conversion
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
