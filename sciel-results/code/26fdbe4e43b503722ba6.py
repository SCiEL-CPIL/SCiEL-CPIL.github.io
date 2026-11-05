"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 80.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # best_size[r] = minimum subset size that sums to r mod MOD
    # best_mask[r] = the mask achieving that minimum size
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD

    # Iterate over all non-empty subsets (only 1023 total, so this is fast)
    for mask in range(1, 1 << N):
        if _elapsed() > _DEADLINE:
            break

        s = 0
        m = mask
        size = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb

        r = s % MOD

        if size < best_size[r]:
            best_size[r] = size
            best_mask[r] = mask

    # Construct rows
    rows = []
    for r in range(MOD):
        if best_mask[r] == 0:
            # Fallback: use full set (should not happen since all residues are covered)
            rows.append(list(range(1, N + 1)))
        else:
            mask = best_mask[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows.append(terms)

    return rows
# EVOLVE-BLOCK-END
