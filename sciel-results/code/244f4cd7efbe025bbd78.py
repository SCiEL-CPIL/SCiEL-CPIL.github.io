def construct():
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Enumerate all non-empty subsets
    # For each subset, compute (size, residue, list_of_denominators)
    subsets = []  # list of (size, residue, terms)
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + inv[x]) % 101
        subsets.append((len(terms), s, terms))
    
    # Find minimum k such that all 101 residues are covered by subsets of size <= k
    for k in range(1, 11):
        covered = set()
        for size, residue, terms in subsets:
            if size <= k:
                covered.add(residue)
        if len(covered) == 101:
            break
    
    # Assign each residue to a subset of size <= k (prefer smallest size, then lexicographic)
    # For each residue, collect all subsets of size <= k that cover it
    residue_to_subsets = {r: [] for r in range(101)}
    for size, residue, terms in subsets:
        if size <= k:
            residue_to_subsets[residue].append((size, terms))
    
    # For each residue, pick the subset with smallest size (tie-break: lexicographic)
    result = []
    for r in range(101):
        candidates = residue_to_subsets[r]
        candidates.sort(key=lambda x: (x[0], x[1]))
        result.append(candidates[0][1])
    
    return result
