import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: use 80s limit for safety before 90s kill
    start_time = time.time()
    deadline = start_time + 80.0

    p = 101
    n = 10

    # Modular inverses of 1..10 modulo 101
    inverses = [0] * (n + 1)
    for i in range(1, n + 1):
        inverses[i] = pow(i, -1, p)

    # Precompute all non-empty subsets: mask -> (size, residue, terms)
    # We'll group by residue to find the minimum size for each residue
    from collections import defaultdict
    residue_masks = defaultdict(list)

    for mask in range(1, 1 << n):
        size = 0
        s = 0
        terms = []
        for i in range(1, n + 1):
            if mask & (1 << (i - 1)):
                size += 1
                s = (s + inverses[i]) % p
                terms.append(i)
        residue = s % p
        residue_masks[residue].append((size, mask, terms))

    # For each residue, sort by size (ascending) so index 0 is the smallest
    for r in residue_masks:
        residue_masks[r].sort(key=lambda x: x[0])

    # Find the minimum subset size required for each residue
    # If a residue is not achievable (shouldn't happen), we need a fallback
    min_size_for_residue = {}
    for r in range(p):
        if r in residue_masks and len(residue_masks[r]) > 0:
            min_size_for_residue[r] = residue_masks[r][0][0]
        else:
            # This should not happen for p=101, n=10, but handle gracefully
            min_size_for_residue[r] = n  # worst case

    # The optimal maximum row length is the max of these independent minima
    k_opt = max(min_size_for_residue.values())

    # Construct the result: for each residue r, pick the subset with minimum size
    # This achieves max row length = k_opt, which is optimal
    result = []
    for r in range(p):
        if r in residue_masks and len(residue_masks[r]) > 0:
            size, mask, terms = residue_masks[r][0]
            result.append(terms)
        else:
            # Fallback: find any subset that sums to r (should not happen)
            found = False
            for mask in range(1, 1 << n):
                s = 0
                terms = []
                for i in range(1, n + 1):
                    if mask & (1 << (i - 1)):
                        s = (s + inverses[i]) % p
                        terms.append(i)
                if s == r:
                    result.append(terms)
                    found = True
                    break
            if not found:
                # Absolute fallback: use a single element (may not sum to r, but avoids crash)
                # In practice, this path is never taken for p=101, n=10
                result.append([1])

    # Verification pass (cheap, ensures correctness)
    for r, terms in enumerate(result):
        if len(terms) == 0:
            raise ValueError(f"Row {r} is empty")
        if len(set(terms)) != len(terms):
            raise ValueError(f"Row {r} has duplicate denominators")
        if any(x < 1 or x > n for x in terms):
            raise ValueError(f"Row {r} has invalid denominator")
        res = sum(inverses[x] for x in terms) % p
        if res != r:
            raise ValueError(f"Row {r} sums to {res} instead of {r}")

    return result
# EVOLVE-BLOCK-END
