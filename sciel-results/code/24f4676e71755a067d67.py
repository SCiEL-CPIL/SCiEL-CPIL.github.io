import time

def construct():
    deadline = time.time() + 85.0
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # Enumerate all non-empty subsets, store (size, residue, terms)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        size = len(terms)
        residue = sum(inv[x] for x in terms) % 101
        all_subsets.append((size, residue, terms))
    
    # Sort by size
    all_subsets.sort(key=lambda x: x[0])
    
    # For each k, check if all residues are covered by subsets of size <= k
    # We want the smallest k such that every residue 0..100 is hit
    
    # Build: for each residue, the smallest subset (by size, then any) that hits it
    # But we want to minimize max size, so find smallest k where coverage is complete
    
    # For efficiency, precompute for each residue the minimum size subset that hits it
    min_size_for_residue = [11] * 101  # 11 = impossible
    best_terms_for_residue = [None] * 101
    
    for size, residue, terms in all_subsets:
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            best_terms_for_residue[residue] = terms
    
    # The answer is max(min_size_for_residue[r] for r in range(101))
    # But we need to verify all are covered
    max_size = 0
    for r in range(101):
        if min_size_for_residue[r] == 11:
            # This shouldn't happen since we have 1023 subsets for 101 residues
            raise RuntimeError("Residue not covered")
        max_size = max(max_size, min_size_for_residue[r])
    
    # Build the result
    result = [best_terms_for_residue[r] for r in range(101)]
    
    return result
