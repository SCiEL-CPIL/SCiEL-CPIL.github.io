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

    # For each residue r, collect all masks (subsets) that sum to r mod MOD
    # We want to find the minimum k such that there exist masks m_0, ..., m_100
    # with |m_i| <= k for all i, and all residues covered.
    # Since denominators can repeat across rows, we just need for each residue r
    # a subset of size <= k that sums to r.
    # So the answer is: find the minimum k such that every residue r in 0..100
    # has at least one subset of size <= k summing to r mod 101.

    # Precompute: for each mask, its sum mod MOD and its size
    mask_sum = [0] * (1 << N)
    mask_size = [0] * (1 << N)
    for mask in range(1, 1 << N):
        lsb = mask & (-mask)
        x = lsb.bit_length()
        prev = mask - lsb
        mask_sum[mask] = (mask_sum[prev] + inv[x]) % MOD
        mask_size[mask] = mask_size[prev] + 1

    # For each residue, find the minimum subset size that achieves it
    min_size_for_residue = [N + 1] * MOD
    for mask in range(1, 1 << N):
        r = mask_sum[mask]
        sz = mask_size[mask]
        if sz < min_size_for_residue[r]:
            min_size_for_residue[r] = sz

    # The maximum over all residues of min_size_for_residue[r] is the answer
    max_min_size = max(min_size_for_residue)

    # Now construct: for each residue r, pick a mask of size <= max_min_size
    # that sums to r. We already have the minimum, so we pick that.
    best_mask = [0] * MOD
    for mask in range(1, 1 << N):
        r = mask_sum[mask]
        sz = mask_size[mask]
        if sz < min_size_for_residue[r]:
            # This shouldn't happen since we already computed min, but just in case
            min_size_for_residue[r] = sz
            best_mask[r] = mask
        elif sz == min_size_for_residue[r] and best_mask[r] == 0:
            best_mask[r] = mask

    # For residues where best_mask is still 0 (shouldn't happen), find any valid mask
    for r in range(MOD):
        if best_mask[r] == 0:
            for mask in range(1, 1 << N):
                if mask_sum[mask] == r:
                    best_mask[r] = mask
                    break

    # Construct rows
    rows = []
    for r in range(MOD):
        mask = best_mask[r]
        if mask == 0:
            # Fallback: use full set
            rows.append(list(range(1, N + 1)))
        else:
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows.append(terms)

    return rows
# EVOLVE-BLOCK-END
