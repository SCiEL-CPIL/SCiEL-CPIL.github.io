def construct():
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # For each residue, find the minimum subset size
    min_size = [11] * 101  # 11 is "infinity" (max possible is 10)
    best_subset = [None] * 101
    
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        # If we already have a subset of this size or smaller for all residues, 
        # we can skip larger sizes, but let's just compute all
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % 101
        if size < min_size[s]:
            min_size[s] = size
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            best_subset[s] = terms
    
    return [best_subset[r] for r in range(101)]
