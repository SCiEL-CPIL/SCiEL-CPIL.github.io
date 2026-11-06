import time

def construct():
    start_time = time.time()
    deadline = start_time + 80

    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))

    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)

    max_mask = 1 << MAX_DENOM

    # Precompute sum and length for all masks using DP
    mask_sum = [0] * max_mask
    mask_len = [0] * max_mask
    for mask in range(1, max_mask):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        mask_sum[mask] = (mask_sum[prev] + inverses[idx]) % MOD
        mask_len[mask] = mask_len[prev] + 1

    # Group masks by their residue, sorted by length then by mask value
    # We want the shortest subset for each residue
    masks_by_residue = [[] for _ in range(MOD)]
    for mask in range(1, max_mask):
        s = mask_sum[mask]
        masks_by_residue[s].append(mask)

    # For each residue, pick the mask with minimum length
    best_mask = [None] * MOD
    best_len = [MAX_DENOM + 1] * MOD
    for r in range(MOD):
        if masks_by_residue[r]:
            # Find the mask with minimum length
            min_len = MAX_DENOM + 1
            best = None
            for m in masks_by_residue[r]:
                l = mask_len[m]
                if l < min_len:
                    min_len = l
                    best = m
            if best is not None:
                best_mask[r] = best
                best_len[r] = min_len

    # Fallback for any uncovered residues
    for r in range(MOD):
        if best_mask[r] is None:
            best_mask[r] = 1
            best_len[r] = 1

    # Construct the final result
    results = [None] * MOD
    for r in range(MOD):
        mask = best_mask[r]
        combo = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            combo.append(denominators[idx])
            m ^= lsb
        results[r] = combo

    return results
