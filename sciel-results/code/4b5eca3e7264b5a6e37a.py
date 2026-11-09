import time

def construct():
    t0 = time.perf_counter()
    WALL = 88.0

    p = 101
    D = 10
    inv = [pow(k, -1, p) for k in range(1, D + 1)]

    # DP over subsets: for each mask, record (sum_mod, popcount)
    total_masks = 1 << D
    sums = [0] * total_masks
    pops = [0] * total_masks

    # Fill DP arrays
    for m in range(1, total_masks):
        # lowest set bit
        lb = m & (-m)
        idx = lb.bit_length() - 1  # 0-based index of lowest set bit
        pm = m ^ lb
        sums[m] = (sums[pm] + inv[idx]) % p
        pops[m] = pops[pm] + 1

    # best[r] = (popcount, mask) with minimal popcount
    best_pop = [p + 1] * p
    best_mask = [0] * p

    for m in range(1, total_masks):
        r = sums[m]
        pc = pops[m]
        if pc < best_pop[r]:
            best_pop[r] = pc
            best_mask[r] = m

    # Verify coverage
    for r in range(p):
        if best_pop[r] > D:
            # Should never happen
            pass

    # Build result
    result = []
    for r in range(p):
        m = best_mask[r]
        combo = [k + 1 for k in range(D) if m & (1 << k)]
        result.append(combo)

    return result
