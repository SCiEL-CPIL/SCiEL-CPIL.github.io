def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    P = 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]

    # Enumerate all non-empty subsets, group by size
    # subsets_by_size[sz] = list of (residue, terms)
    subsets_by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets_by_size[len(terms)].append((s, terms))

    # Find minimum L such that all 101 residues are covered by subsets of size <= L
    # Each residue just needs at least one subset of size <= L producing it.
    # Subsets CAN be reused across different rows.
    assignment = {}
    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms
        if len(assignment) == P:
            # All residues covered with max row length L
            return [assignment[r] for r in range(P)]

    # Fallback: should never reach here since L=10 includes all subsets
    # But if we do, use greedy assignment
    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms

    return [assignment.get(r, [1]) for r in range(P)]
