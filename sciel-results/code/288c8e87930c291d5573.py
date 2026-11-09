import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # best_residue[r] will hold the shortest list for residue r
    # Initialize with None
    best_residue = [None] * 101
    best_len = [float('inf')] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # mask represents the subset
    for mask in range(1, 1 << 10):
        # Optional: check time
        # if time.time() > deadline: break
        
        # Calculate sum and length
        s = 0
        length = 0
        m = mask
        i = 1
        while m:
            if m & 1:
                s += inv[i]
                length += 1
            i += 1
            m >>= 1
            
        r = s % 101
        
        if length < best_len[r]:
            # Reconstruct the list
            terms = []
            m2 = mask
            i2 = 1
            while m2:
                if m2 & 1:
                    terms.append(i2)
                i2 += 1
                m2 >>= 1
            best_residue[r] = terms
            best_len[r] = length
            
    # Construct the final result
    # All residues should be covered. If not, we might need a fallback, 
    # but the problem statement implies a covering exists.
    # The seed program returns [representations[r] for r in range(101)], 
    # which would fail if any is None. 
    # Given the seed works, all are covered.
    
    return best_residue
