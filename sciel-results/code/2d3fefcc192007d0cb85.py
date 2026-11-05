# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 82.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    """
    Constructs 101 lists of denominators (1-10) such that the sum of their
    modular inverses (mod 101) equals the list index.
    Minimizes the maximum length of any list.
    
    Uses DP over bitmasks to compute subset sums efficiently, then for each
    residue finds the minimum-size subset that achieves it.
    """
    n = 10
    mod = 101
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[i] = pow(i, -1, mod)
    
    # DP: for each mask, compute sum mod 101 and size
    # mask ranges from 0 to (1<<n)-1
    num_masks = 1 << n
    subset_sum = [0] * num_masks
    subset_size = [0] * num_masks
    
    # For each mask, find the lowest set bit to enable DP recurrence
    for mask in range(1, num_masks):
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length() - 1  # 0-indexed: 0 for bit 0, 1 for bit 1, etc.
        prev_mask = mask ^ lsb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % mod
        subset_size[mask] = subset_size[prev_mask] + 1
    
    # For each residue, find the minimum-size subset (and among ties, smallest mask)
    min_size = [n + 1] * mod  # Initialize to infinity
    best_mask = [0] * mod
    
    for mask in range(1, num_masks):
        r = subset_sum[mask]
        size = subset_size[mask]
        if size < min_size[r]:
            min_size[r] = size
            best_mask[r] = mask
        elif size == min_size[r] and mask < best_mask[r]:
            best_mask[r] = mask
    
    # Construct the result rows
    rows = []
    for r in range(mod):
        mask = best_mask[r]
        terms = []
        for i in range(1, n + 1):
            if (mask >> (i - 1)) & 1:
                terms.append(i)
        rows.append(terms)
    
    return rows
# EVOLVE-BLOCK-END
