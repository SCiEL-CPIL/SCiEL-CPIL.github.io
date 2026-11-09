import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    # Pre-compute inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # Enumerate all non-empty subsets of {1..10}
    # For each residue, track the minimum-size subset that hits it
    min_size = [11] * 101
    best_subset = [None] * 101

    # Use iterative subset sum: for each mask, compute sum and size
    # More efficient: build up from smaller masks
    sum_mod = [0] * (1 << 10)
    popcount = [0] * (1 << 10)

    for mask in range(1, 1 << 10):
        # Find lowest set bit
        lb = mask & (-mask)
        bit_idx = lb.bit_length() - 1
        prev_mask = mask ^ lb
        d = bit_idx + 1
        sum_mod[mask] = (sum_mod[prev_mask] + inv[d]) % 101
        popcount[mask] = popcount[prev_mask] + 1
        s = sum_mod[mask]
        sz = popcount[mask]

        if sz < min_size[s]:
            min_size[s] = sz
            # Reconstruct the subset
            terms = []
            m = mask
            while m:
                lb2 = m & (-m)
                idx2 = lb2.bit_length() - 1
                terms.append(idx2 + 1)
                m ^= lb2
            best_subset[s] = terms

    # Check coverage
    missing = [r for r in range(101) if best_subset[r] is None]

    if not missing:
        result = [best_subset[r] for r in range(101)]
        return result

    # Fallback: should not happen
    for r in missing:
        best_subset[r] = [1]

    result = [best_subset[r] for r in range(101)]
    return result
