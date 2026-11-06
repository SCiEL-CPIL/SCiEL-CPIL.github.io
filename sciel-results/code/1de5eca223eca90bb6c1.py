import time

def construct():
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)
        
    # Total number of subsets is 2^10 = 1024
    # We use bitmask to represent subsets
    
    max_mask = 1 << MAX_DENOM
    subset_sums = [0] * max_mask
    subset_lens = [0] * max_mask
    
    for mask in range(1, max_mask):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        subset_sums[mask] = (subset_sums[prev_mask] + inverses[idx]) % MOD
        subset_lens[mask] = subset_lens[prev_mask] + 1
        
    # Determine the minimum possible maximum length (L_star)
    min_len_for_residue = [11] * MOD
    
    for mask in range(1, max_mask):
        r = subset_sums[mask]
        l = subset_lens[mask]
        if l < min_len_for_residue[r]:
            min_len_for_residue[r] = l
            
    L_star = max(min_len_for_residue)
    
    if L_star > MAX_DENOM:
        L_star = MAX_DENOM
        
    # Construct the result
    # Pick the shortest subset for each residue.
    # Tie-breaker: smallest mask.
    
    current_best_len = [11] * MOD
    current_best_mask = [0] * MOD
    
    for mask in range(1, max_mask):
        r = subset_sums[mask]
        l = subset_lens[mask]
        
        if l < current_best_len[r]:
            current_best_len[r] = l
            current_best_mask[r] = mask
        elif l == current_best_len[r]:
            if mask < current_best_mask[r]:
                current_best_mask[r] = mask
                
    results = [None] * MOD
    for r in range(MOD):
        mask = current_best_mask[r]
        if mask == 0:
            # Fallback
            for m in range(1, max_mask):
                if subset_sums[m] == r:
                    mask = m
                    break
            if mask == 0:
                results[r] = [1]
                continue
        
        subset = []
        for i in range(MAX_DENOM):
            if mask & (1 << i):
                subset.append(denominators[i])
        results[r] = subset
        
    return results
