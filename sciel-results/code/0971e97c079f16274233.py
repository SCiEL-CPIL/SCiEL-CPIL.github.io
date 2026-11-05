"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # For each residue, store the minimum subset size and corresponding mask
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD

    # Precompute sum and size for every mask
    mask_sum = [0] * (1 << N)
    mask_size = [0] * (1 << N)
    for mask in range(1, 1 << N):
        # Use lowest set bit to compute sum and size
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length()  # 1-indexed
        prev = mask ^ lsb
        mask_sum[mask] = mask_sum[prev] + inv[bit_idx]
        mask_size[mask] = mask_size[prev] + 1

    # Iterate over all non-empty subsets
    for mask in range(1, 1 << N):
        if _elapsed() > _DEADLINE:
            break

        r = mask_sum[mask] % MOD
        sz = mask_size[mask]

        if sz < best_size[r]:
            best_size[r] = sz
            best_mask[r] = mask

    # Check if all residues are covered
    uncovered = [r for r in range(MOD) if best_mask[r] == 0]

    if not uncovered:
        # All residues covered with minimum subsets
        rows = []
        for r in range(MOD):
            mask = best_mask[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows.append(terms)
        return rows

    # Fallback: try to find solutions for uncovered residues using larger subsets
    # This shouldn't happen for MOD=101, N=10, but handle gracefully
    for r in uncovered:
        # Find any subset that sums to r
        for mask in range(1, 1 << N):
            if mask_sum[mask] % MOD == r:
                best_mask[r] = mask
                best_size[r] = mask_size[mask]
                break

    rows = []
    for r in range(MOD):
        mask = best_mask[r]
        terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        rows.append(terms)

    return rows
# EVOLVE-BLOCK-END
