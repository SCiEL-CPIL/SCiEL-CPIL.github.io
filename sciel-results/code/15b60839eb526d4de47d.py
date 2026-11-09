def construct():

    p = 101
    n = 10
    denominators = list(range(1, n + 1))

    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]

    # Bitmask DP
    total_masks = 1 << n
    subset_sums = [0] * total_masks

    # Precompute popcounts
    popcounts = [0] * total_masks
    for mask in range(1, total_masks):
        popcounts[mask] = popcounts[mask >> 1] + (mask & 1)

    # Compute subset sums
    for mask in range(1, total_masks):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        subset_sums[mask] = (subset_sums[prev_mask] + inverses[idx]) % p

    # Find min popcount subset for each residue
    # best_size[r] = min size, best_mask[r] = corresponding mask
    INF = 11
    best_size = [INF] * p
    best_mask = [0] * p

    for mask in range(1, total_masks):
        residue = subset_sums[mask]
        size = popcounts[mask]
        if size < best_size[residue]:
            best_size[residue] = size
            best_mask[residue] = mask

    # Check coverage
    # If any best_size is INF, it means uncovered.
    # But we know it's covered.
    # Fallback just in case
    full_mask = total_masks - 1
    for r in range(p):
        if best_size[r] == INF:
            best_size[r] = n
            best_mask[r] = full_mask

    # Build result
    result = []
    for r in range(p):
        mask = best_mask[r]
        combo = []
        for i in range(n):
            if mask & (1 << i):
                combo.append(i + 1)
        result.append(combo)

    return result
