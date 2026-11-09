import time

def construct():
    deadline = time.time() + 85.0
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # We want to find the smallest K such that every residue 0..100 
    # is covered by a subset of size <= K.
    # Then we assign to each residue one such subset.
    
    # Generate all subsets and their sums
    # subsets[residue] = list of (size, terms)
    subsets_by_residue = {r: [] for r in range(101)}
    
    # Iterate over all non-empty subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                terms.append(d)
                s = (s + inv[d]) % 101
        size = len(terms)
        subsets_by_residue[s].append((size, terms))
        
    # Find the minimum K
    # For each residue, find the min size
    min_sizes = [11] * 101
    best_terms = [None] * 101
    
    for r in range(101):
        if not subsets_by_residue[r]:
            # Should not happen
            continue
        # Find min size in this list
        min_size = 11
        best_t = None
        for size, terms in subsets_by_residue[r]:
            if size < min_size:
                min_size = size
                best_t = terms
        min_sizes[r] = min_size
        best_terms[r] = best_t
        
    max_len = max(min_sizes)
    
    # The result is the list of terms for each residue
    result = [best_terms[r] for r in range(101)]
    
    return result
