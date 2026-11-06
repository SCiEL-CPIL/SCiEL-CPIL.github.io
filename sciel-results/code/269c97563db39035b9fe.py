import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 85.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Build all nonempty subsets grouped by size
    # subsets_by_size[size] = list of (residue, terms)
    subsets_by_size = defaultdict(list)

    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        size = len(terms)
        subsets_by_size[size].append((s, terms))

    # Find minimum L such that subsets of sizes 1..L cover all residues 0..100
    # Greedy: for each residue, assign the first subset (smallest size first) that hits it
    assignment = {}

    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms

        if len(assignment) == P:
            # All residues covered with max size L
            return [assignment[r] for r in range(P)]

        if time.time() > deadline:
            # Safety: should never trigger for this small problem
            break

    # Fallback (should be unreachable)
    return [assignment.get(r, [1]) for r in range(P)]
