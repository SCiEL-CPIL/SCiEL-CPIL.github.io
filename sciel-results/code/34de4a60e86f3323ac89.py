def local_search(S, Iso):
    S = list(S)
    S_idx = [PT_IDX[p] for p in S]
    S_bits = 0
    for pi in S_idx:
        S_bits |= (1 << pi)
    improved = True
    while improved:
        improved = False
        essential_mask = 0
        for q in ALL_POINTS:
            qi = PT_IDX[q]
            if (S_bits >> qi) & 1:
                continue
            base_q = qi*NP
            cover_count = 0
            cover_single = -1
            for ai in S_idx:
                if (Iso[base_q + ai] & S_bits) & ~(1 << ai):
                    cover_count += 1
                    if cover_count == 1:
                        cover_single = ai
                    else:
                        break
            if cover_count == 1:
                essential_mask |= (1 << cover_single)
        removable_mask = S_bits & ~essential_mask
        if removable_mask:
            new_S = []
            new_idx = []
            new_bits = 0
            for p, pi in zip(S, S_idx):
                if (removable_mask >> pi) & 1:
                    continue
                new_S.append(p)
                new_idx.append(pi)
                new_bits |= (1 << pi)
            S = new_S
            S_idx = new_idx
            S_bits = new_bits
            improved = True
    return S
