import itertools

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15 for bitmask simplicity, or 1..16?
    # Task says [1, n=16]. Bitmasks usually 0-indexed.
    # Let's map element v (1..16) to bit v-1.
    
    n = 16
    full_mask = (1 << n) - 1
    
    all_masks = []
    mask_to_idx = {}
    
    # Generate combinations of 4 from 0..15
    for comb in itertools.combinations(range(n), 4):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
        
    N = len(all_masks)
    
    # Precompute list of set bits for each mask? 
    # Or just compute on the fly in get_conflicts.
    
    def get_conflict_indices(mask_a, mask_b):
        """
        Returns list of indices k such that (mask_a, mask_b, mask_k) form a sunflower.
        """
        inter = mask_a & mask_b
        popcount_inter = bin(inter).count('1')
        
        # Forbidden elements: (A \ I) U (B \ I)
        # A \ I = A & ~I
        # B \ I = B & ~I
        forbidden = (mask_a & ~inter) | (mask_b & ~inter)
        
        # Free elements: Full \ (Forbidden U I)
        # Note: I is disjoint from Forbidden.
        free_mask = full_mask & ~(forbidden | inter)
        
        # We need to pick (4 - popcount_inter) elements from free_mask
        need = 4 - popcount_inter
        
        if need < 0 or need > bin(free_mask).count('1'):
            return []
            
        # Get indices of set bits in free_mask
        free_bits = []
        m = free_mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            free_bits.append(idx)
            m ^= lsb
            
        conflicts = []
        # Generate combinations of 'need' from free_bits
        for comb in itertools.combinations(free_bits, need):
            y_mask = 0
            for bit_idx in comb:
                y_mask |= (1 << bit_idx)
            
            k_mask = inter | y_mask
            
            # Check if k_mask is a valid set (it should be, as we constructed it from valid universe)
            # And k_mask != mask_a and k_mask != mask_b
            if k_mask == mask_a or k_mask == mask_b:
                continue
                
            idx = mask_to_idx.get(k_mask, -1)
            if idx != -1:
                conflicts.append(idx)
                
        return conflicts

    conflict_counts = [0] * N
    avail_zero = set(range(N))
    chosen = []
    
    while avail_zero:
        # Pick any from avail_zero (all have 0 conflicts with chosen)
        # To be deterministic, we can pick the smallest index
        best = min(avail_zero)
        
        avail_zero.remove(best)
        conflict_counts[best] = -1
        chosen.append(best)
        
        # Update conflict counts for other sets
        best_mask = all_masks[best]
        for old_idx in chosen[:-1]: # Exclude self
            old_mask = all_masks[old_idx]
            conflicts = get_conflict_indices(best_mask, old_mask)
            for k_idx in conflicts:
                if conflict_counts[k_idx] == 0:
                    conflict_counts[k_idx] = 1
                    if k_idx in avail_zero:
                        avail_zero.remove(k_idx)
                else:
                    conflict_counts[k_idx] += 1
                    
    return [sorted([i+1 for i in range(n) if (all_masks[idx] >> i) & 1]) for idx in chosen]
