"""Optimized construction of large 4-uniform sunflower-free family using Min-Degree Greedy."""

def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 85

    # 1. Generate all 4-subsets of [1, 16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    # Map set to index
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    
    # Convert to bitmasks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
        
    masks = [to_mask(s) for s in all_sets]
    mask_to_idx = {m: i for i, m in enumerate(masks)}
    
    # Precompute bits for all 16-bit masks
    def get_bits(m):
        bits = []
        while m:
            lsb = m & -m
            bits.append(lsb)
            m ^= lsb
        return bits

    bits_cache = [None] * (1 << 16)
    for m in range(1 << 16):
        bits_cache[m] = get_bits(m)

    # 2. Precompute conflict masks
    # conflict_mask[i][j] is a bitmask of indices k such that {i, j, k} is a sunflower
    conflict_mask = [[0] * n_total for _ in range(n_total)]
    
    for i in range(n_total):
        mi = masks[i]
        for j in range(i + 1, n_total):
            mj = masks[j]
            
            inter = mi & mj
            union = mi | mj
            
            r_mask = ((1 << 16) - 1) ^ union
            k_size = 4 - inter.bit_count()
            
            if k_size < 0 or k_size > r_mask.bit_count():
                continue
                
            r_bits = bits_cache[r_mask]
            
            # Generate subsets of r_bits of size k_size
            # If k_size is 0, only one subset (empty)
            # If k_size is 1, len(r_bits) subsets
            # etc.
            
            # Optimization: If k_size is small, use combinations
            for combo in combinations(range(len(r_bits)), k_size):
                extra = 0
                for idx in combo:
                    extra |= r_bits[idx]
                
                k_mask = inter | extra
                k_idx = mask_to_idx.get(k_mask, -1)
                
                if k_idx != -1 and k_idx != i and k_idx != j:
                    conflict_mask[i][j] |= (1 << k_idx)
                    conflict_mask[j][i] |= (1 << k_idx)

    # 3. Greedy Algorithm
    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Try multiple restarts
    # Adjust restart count based on time
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        available = (1 << n_total) - 1
        selected = []
        
        while available:
            # Find v in available with min cost
            # Cost of v = bit_count( (OR of conflict_mask[v][w] for w in selected) | (1<<v) & available )
            
            min_cost = n_total + 1
            best_v = -1
            
            # Get list of available indices
            avail_indices = []
            temp_avail = available
            while temp_avail:
                lsb = temp_avail & -temp_avail
                idx = lsb.bit_length() - 1
                avail_indices.append(idx)
                temp_avail ^= lsb
                
            if not avail_indices:
                break
                
            if not selected:
                # Pick random to diversify
                v = rng.choice(avail_indices)
            else:
                # Evaluate each candidate
                for v in avail_indices:
                    mask = 1 << v
                    for w in selected:
                        mask |= conflict_mask[v][w]
                        
                    cost = (mask & available).bit_count()
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_v = v
                        
                v = best_v
            
            # Add v to selected
            selected.append(v)
            
            # Update available
            remove_mask = 1 << v
            for w in selected[:-1]:
                remove_mask |= conflict_mask[v][w]
                
            available &= ~remove_mask
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_family = list(selected)
            
    # Convert indices to sets
    return [sorted(all_sets[i]) for i in best_family]
