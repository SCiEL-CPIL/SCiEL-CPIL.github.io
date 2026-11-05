"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 85.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Precompute all non-empty subsets: (sum_mod, size, mask)
    # There are 2^10 - 1 = 1023 subsets, very fast
    subsets = []
    for mask in range(1, 1 << N):
        s = 0
        size = 0
        m = mask
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s = (s + inv[x]) % MOD
            size += 1
            m -= lsb
        subsets.append((s, size, mask))

    # Group by residue: by_residue[r] = list of (size, mask)
    by_residue = [[] for _ in range(MOD)]
    for s, size, mask in subsets:
        by_residue[s].append((size, mask))

    # Sort each residue's list by size ascending
    for r in range(MOD):
        by_residue[r].sort(key=lambda x: x[0])

    # Find the minimum k such that every residue has at least one subset of size <= k
    # This is the theoretical optimal maximum row length
    def can_cover(k):
        for r in range(MOD):
            if not by_residue[r] or by_residue[r][0][0] > k:
                return False
        return True

    lo, hi = 1, N
    best_k = N
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_cover(mid):
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # For each residue, pick the smallest-size subset (with size <= best_k)
    # Since by_residue[r] is sorted by size, the first entry with size <= best_k is optimal
    result = []
    for r in range(MOD):
        chosen_mask = None
        for size, mask in by_residue[r]:
            if size <= best_k:
                chosen_mask = mask
                break
        if chosen_mask is None:
            # Fallback: should not happen if best_k is correct
            if by_residue[r]:
                chosen_mask = by_residue[r][0][1]
            else:
                chosen_mask = 0

        terms = []
        for x in range(1, N + 1):
            if chosen_mask & (1 << (x - 1)):
                terms.append(x)
        result.append(terms)

    return result

# EVOLVE-BLOCK-END
