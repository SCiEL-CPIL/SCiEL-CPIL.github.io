"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 85.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def _mask_to_terms(mask, n=10):
    return [x for x in range(1, n + 1) if (mask >> (x - 1)) & 1]


def construct():
    n = 10
    mod = 101
    inv = [0] + [pow(x, -1, mod) for x in range(1, n + 1)]
    
    # Precompute sum for each mask efficiently using DP
    # sum_val[mask] = sum of inv[x] for x in mask, mod 101
    total_masks = 1 << n
    sum_val = [0] * total_masks
    popcount = [0] * total_masks
    
    for mask in range(1, total_masks):
        lsb = mask & (-mask)
        x = lsb.bit_length()  # 1-based
        prev = mask ^ lsb
        sum_val[mask] = (sum_val[prev] + inv[x]) % mod
        popcount[mask] = popcount[prev] + 1
    
    # Group masks by popcount
    by_pop = [[] for _ in range(n + 1)]
    for mask in range(1, total_masks):
        by_pop[popcount[mask]].append(mask)
    
    # For each residue, find the smallest-size subset that sums to it
    best_mask = [None] * mod
    for pc in range(1, n + 1):
        if _elapsed() > _DEADLINE:
            break
        for mask in by_pop[pc]:
            r = sum_val[mask]
            if best_mask[r] is None:
                best_mask[r] = mask
    
    # Check if all residues are covered
    if all(m is not None for m in best_mask):
        rows = [_mask_to_terms(best_mask[r]) for r in range(mod)]
        return rows
    
    # Fallback: use any representation
    representations = {}
    for mask in range(1, total_masks):
        r = sum_val[mask]
        if r not in representations:
            representations[r] = mask
    rows = []
    for r in range(mod):
        if r in representations:
            rows.append(_mask_to_terms(representations[r]))
        else:
            rows.append(_mask_to_terms((1 << n) - 1))
    return rows

# EVOLVE-BLOCK-END
