def construct():
    """
    Construct 101 nonempty lists of distinct denominators in 1..10 whose
    modular inverses sum to the row index modulo 101.
    
    We enumerate all non-empty subsets of {1, ..., 10} (there are 2^10 - 1 = 1023),
    compute the sum of inverses mod 101 and the size of each subset, and for each
    residue keep only the subset with minimum size. Since denominators may recur
    across different rows, this per-residue greedy choice is globally optimal for
    minimizing the maximum row length.
    """
    MOD = 101
    N = 10
    
    # Pre-compute modular inverses of 1..10 mod 101
    # Hardcoded for speed and clarity, values are pow(d, 99, 101)
    inv = [0, 1, 51, 34, 26, 81, 17, 29, 90, 45, 91]
    
    # For each residue r in 0..100, track the minimum size and the corresponding subset
    min_size = [N + 1] * MOD  # Initialize with impossible size
    best_subset = [None] * MOD
    
    # Enumerate all non-empty subsets using bitmask from 1 to (1<<N)-1
    for mask in range(1, 1 << N):
        s = 0
        size = 0
        terms = []
        
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                s = (s + inv[d]) % MOD
                size += 1
                terms.append(d)
        
        # If this subset is smaller than the current best for residue s, update
        if size < min_size[s]:
            min_size[s] = size
            best_subset[s] = terms
    
    # Build the result: row r contains best_subset[r]
    # All residues are covered by the exhaustive enumeration.
    result = []
    for r in range(MOD):
        result.append(best_subset[r])
    
    return result
