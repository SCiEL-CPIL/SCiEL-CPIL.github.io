import time

def construct():
    start_time = time.time()
    LIMIT = 85.0  # 90s wall-clock, safety margin

    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]

    # For each residue, find the minimum-size subset that sums to it
    # best[r] = (min_size, mask) for residue r
    best = [None] * p

    # Iterate all non-empty subsets (1023 total)
    for mask in range(1, 1 << n):
        s = 0
        sz = 0
        for i in range(n):
            if mask & (1 << i):
                s += inverses[i]
                sz += 1
        r = s % p
        if best[r] is None or sz < best[r][0]:
            best[r] = (sz, mask)

    # Verify all residues are covered
    for r in range(p):
        if best[r] is None:
            raise RuntimeError(f"Residue {r} not covered")

    # Build result
    result = []
    for r in range(p):
        sz, mask = best[r]
        combo = [denominators[i] for i in range(n) if mask & (1 << i)]
        result.append(combo)

    # Verify correctness
    for r in range(p):
        s = sum(inverses[d - 1] for d in result[r])
        if s % p != r:
            raise AssertionError(f"Residue {r}: got {s % p}")

    return result
