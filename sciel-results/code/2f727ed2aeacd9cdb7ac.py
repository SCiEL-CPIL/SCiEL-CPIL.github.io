import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Modular inverses of 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # Precompute all non-empty subsets
    # Each subset is represented by a mask
    # We store: (size, residue, mask)
    subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                residue = (residue + inv[x]) % 101
        subsets.append((size, residue, mask))
        
    # Find minimal K such that all residues 0..100 are covered by subsets of size <= K
    # We can iterate K from 1 to 10
    # For efficiency, we can just group subsets by size and accumulate coverage
    
    # coverage[r] = True if residue r is covered by some subset of size <= current_K
    coverage = [False] * 101
    
    # We want to find the smallest K.
    # Let's sort subsets by size to process them in order? 
    # Or just loop K from 1 to 10 and check coverage.
    
    # Actually, we can just compute for each residue the minimum size of a subset that produces it.
    min_size_for_residue = [11] * 101  # 11 is larger than max possible size (10)
    
    for size, residue, mask in subsets:
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            
    # The optimal K is the maximum of min_size_for_residue
    K_opt = max(min_size_for_residue)
    
    # Check if any residue is uncovered (should not happen, but good for safety)
    if K_opt > 10:
        # Fallback: just use any subset, though this shouldn't happen for p=101, n=10
        # We can just pick the first available subset for each residue
        pass
        
    # Construct the result
    # We need to pick a specific subset for each residue.
    # To be deterministic and simple, we can iterate through subsets again and assign the first one found for each residue.
    # However, we must ensure the assigned subset has size <= K_opt.
    # Actually, since K_opt is the max of the min sizes, any subset with size <= K_opt is valid.
    # We just need *one* valid subset per residue.
    
    assigned = [None] * 101
    
    # To ensure we pick subsets that are actually small (<= K_opt), we can filter.
    # But since min_size_for_residue[r] <= K_opt, there exists at least one subset for r with size <= K_opt.
    # We can just pick the one with the smallest size for that residue, or the first one found.
    # Let's pick the one with the smallest size to be "minimal" in a local sense, 
    # although the objective is only the global max.
    
    # Let's create a mapping: residue -> best_mask (smallest size, then lowest mask for tie-breaking)
    best_mask_for_residue = [None] * 101
    best_size_for_residue = [11] * 101
    
    for size, residue, mask in subsets:
        if size < best_size_for_residue[residue]:
            best_size_for_residue[residue] = size
            best_mask_for_residue[residue] = mask
        elif size == best_size_for_residue[residue]:
            if mask < best_mask_for_residue[residue]:
                best_mask_for_residue[residue] = mask
                
    # Build the final lists
    result = []
    for r in range(101):
        mask = best_mask_for_residue[r]
        if mask is None:
            # This should not happen, but handle gracefully
            terms = [1]
        else:
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
        result.append(terms)
        
    return result
