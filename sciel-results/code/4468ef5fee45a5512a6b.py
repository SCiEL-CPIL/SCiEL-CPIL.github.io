COMB4 = list(itertools.combinations(range(N), 4))

def has_q3(adj):
    for A in COMB4:
        a0, a1, a2, a3 = A
        mask_A = (1 << a0) | (1 << a1) | (1 << a2) | (1 << a3)
        candidates = []
        for v in range(N):
            if v == a0 or v == a1 or v == a2 or v == a3:
                continue
            if (adj[v] & mask_A).bit_count() == 3:
                candidates.append(v)
        if len(candidates) < 4:
            continue
        for B in itertools.combinations(candidates, 4):
            b0, b1, b2, b3 = B
            mask_B = (1 << b0) | (1 << b1) | (1 << b2) | (1 << b3)
            if (adj[a0] & mask_B).bit_count() == 3 and \
               (adj[a1] & mask_B).bit_count() == 3 and \
               (adj[a2] & mask_B).bit_count() == 3 and \
               (adj[a3] & mask_B).bit_count() == 3:
                return True
    return False
