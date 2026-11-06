import time

def construct():
    P = 101

    # Precompute modular inverses for 1..10 mod 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]

    # Enumerate all non-empty subsets of {1..10}, grouped by size
    # subsets_by_size[size] = list of (residue, terms_tuple)
    subsets_by_size = [[] for _ in range(11)]

    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets_by_size[len(terms)].append((s, tuple(terms)))

    # Greedy: for L = 1, 2, ..., assign first-seen subset per residue
    assignment = {}

    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms

        if len(assignment) == P:
            return [assignment[r] for r in range(P)]

    # Fallback (should never be reached)
    return [assignment.get(r, (1,)) for r in range(P)]
