import time

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    P = 101
    D = 10
    elements = list(range(1, D + 1))

    # Precompute modular inverses
    inv = [0] * (D + 1)
    for i in range(1, D + 1):
        inv[i] = pow(i, -1, P)

    # Enumerate all nonempty subsets using bitmask, compute residue and size
    # Store as: subsets_by_size[size].append((residue, list_of_elements))
    subsets_by_size = [[] for _ in range(D + 1)]

    for mask in range(1, 1 << D):
        res = 0
        elems = []
        for i in range(D):
            if mask & (1 << i):
                x = i + 1
                res = (res + inv[x]) % P
                elems.append(x)
        sz = len(elems)
        subsets_by_size[sz].append((res, elems))

    # Greedily assign shortest subset to each residue
    # Process by increasing length
    results = [None] * P

    for length in range(1, D + 1):
        if time.time() - t0 > deadline:
            break
        for res, elems in subsets_by_size[length]:
            if results[res] is None:
                results[res] = elems

        # Check if all residues are covered
        if all(r is not None for r in results):
            break

    # If any residue is still None (shouldn't happen), fill with any valid subset
    if any(r is None for r in results):
        # Fallback: use any subset
        for mask in range(1, 1 << D):
            res = 0
            elems = []
            for i in range(D):
                if mask & (1 << i):
                    x = i + 1
                    res = (res + inv[x]) % P
                    elems.append(x)
            if results[res] is None:
                results[res] = elems
        # If still not covered (impossible), use a default
        for r in range(P):
            if results[r] is None:
                results[r] = [1]

    return results
