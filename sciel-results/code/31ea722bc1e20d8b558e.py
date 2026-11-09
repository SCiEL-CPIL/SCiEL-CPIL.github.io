import time

def construct():
    t0 = time.perf_counter()
    WALL = 88.0

    p = 101
    D = 10
    # Compute modular inverses
    inv = []
    for k in range(1, D + 1):
        inv.append(pow(k, -1, p))

    # DP over subsets: for each mask, record sum_mod and popcount
    total_masks = 1 << D
    sums = [0] * total_masks
    pops = [0] * total_masks

    # Fill DP arrays
    for m in range(1, total_masks):
        lb = m & (-m)
        idx = lb.bit_length() - 1
        pm = m ^ lb
        sums[m] = (sums[pm] + inv[idx]) % p
        pops[m] = pops[pm] + 1

    # For each residue, find the subset with minimal popcount
    best_pop = [D + 1] * p
    best_mask = [0] * p

    for m in range(1, total_masks):
        r = sums[m]
        pc = pops[m]
        if pc < best_pop[r]:
            best_pop[r] = pc
            best_mask[r] = m

    # Verify all residues are covered
    for r in range(p):
        if best_pop[r] > D:
            raise ValueError(f"Residue {r} not covered")

    # Build result
    result = []
    for r in range(p):
        m = best_mask[r]
        combo = [k + 1 for k in range(D) if m & (1 << k)]
        result.append(combo)

    # Validate: each row has distinct denominators in 1..10, sum of inverses == r mod 101
    for r in range(p):
        row = result[r]
        if len(row) != len(set(row)):
            raise ValueError(f"Row {r} has duplicate denominators")
        s = 0
        for d in row:
            s = (s + pow(d, -1, p)) % p
        if s != r:
            raise ValueError(f"Row {r} sum {s} != {r}")

    return result
