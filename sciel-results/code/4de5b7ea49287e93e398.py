def _hitting_number(triples):
    n = len(triples)
    if n == 0:
        return 0
    full = (1 << n) - 1
    cover = [0]*16
    for i, t in enumerate(triples):
        bit = 1 << i
        for p in t:
            cover[p] |= bit
    # DP over subsets
    best = 16
    # mask_or[s] for s in 0..2^16-1
    # but 2^16 = 65536, storing is fine
    # compute incrementally
    or_mask = [0]*65536
    pop = [0]*65536
    for s in range(1, 65536):
        lb = s & (-s)
        p = (lb.bit_length() - 1)
        prev = s ^ lb
        or_mask[s] = or_mask[prev] | cover[p]
        pop[s] = pop[prev] + 1
        if or_mask[s] == full and pop[s] < best:
            best = pop[s]
    return best
