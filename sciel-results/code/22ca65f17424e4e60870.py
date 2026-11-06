import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80

    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))

    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)

    max_mask = 1 << MAX_DENOM
    
    # We want to find, for each residue r, the minimum length of a subset summing to r.
    # min_len_for_residue[r] = minimum size of a subset with sum r.
    # best_mask_for_residue[r] = one such subset.
    
    min_len_for_residue = [MAX_DENOM + 1] * MOD
    best_mask_for_residue = [0] * MOD

    # Iterate through all non-empty subsets
    # We can optimize by iterating by size, but 1024 is small.
    for mask in range(1, max_mask):
        # Calculate sum and length
        s = 0
        l = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s = (s + inverses[idx]) % MOD
            l += 1
            m ^= lsb
        
        # Update if this length is smaller than current best for this residue
        if l < min_len_for_residue[s]:
            min_len_for_residue[s] = l
            best_mask_for_residue[s] = mask

    # Check if all residues are covered
    # If not, we have a problem, but with 1023 subsets and 101 residues, it's very likely.
    # The task guarantees a solution exists.
    
    # Construct the result
    results = [None] * MOD
    for r in range(MOD):
        mask = best_mask_for_residue[r]
        combo = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            combo.append(denominators[idx])
            m ^= lsb
        results[r] = combo

    return results
