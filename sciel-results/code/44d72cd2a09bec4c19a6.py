"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_masks():
    """
    For each integer m in [1, 1000024], compute a 9-bit mask where bit j
    is set if the j-th prime (2,3,5,7,11,13,17,19,23) divides m.
    Returns a numpy uint8 array of length 1000025 (indices 0..1000024).
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    size = 1000025
    masks = np.zeros(size, dtype=np.uint8)
    
    for j, p in enumerate(primes):
        # For multiples of p, set bit j
        # Use slicing for speed
        masks[p::p] |= (1 << j)
    
    return masks

def _sliding_window_min_distinct(masks, window_size=24):
    """
    Find n in [1000, 1000000] minimizing the number of distinct prime factors
    (from primes < 24) across the window [n+1, n+24].
    
    We use a frequency array of size 512 (all possible 9-bit masks) to track
    which masks appear in the current window. The OR of all masks in the window
    gives the set of distinct primes. We count bits in that OR.
    
    Actually, a simpler approach: for each position, we need the bitwise OR of
    masks[n+1] through masks[n+24]. We can use a prefix-OR-like approach, but
    OR isn't invertible. So we use the frequency method:
    
    For each mask value (0..511), track how many times it appears in the window.
    The OR of all masks in the window = OR of all mask values with freq > 0.
    
    But computing OR of up to 24 values each step is O(24) which is fine.
    
    Alternative: precompute for each starting position the OR. Since window is 24,
    we can just do it directly with numpy.
    """
    n_max = 1000000
    n_min = 1000
    
    # masks is indexed such that masks[m] is the mask for integer m
    # We need OR of masks[n+1], masks[n+2], ..., masks[n+24]
    
    # For each n from n_min to n_max, compute OR of masks[n+1:n+25]
    # This is a sliding window OR.
    
    # Efficient approach: use the fact that we can maintain a count array.
    # But let's try a direct numpy approach first for clarity, then optimize.
    
    # Direct approach: for each n, OR 24 values. That's 1000000 * 24 operations.
    # With numpy, we can vectorize this.
    
    # Create an array of starting indices
    starts = np.arange(n_min, n_max + 1)
    
    # For each offset 0..23, we need masks[starts + 1 + offset]
    # Then OR them all together.
    
    # This creates a (num_starts, 24) array which might be large.
    # num_starts = 999001, 24 elements each, uint8 = ~24MB. Fine.
    
    # But let's do it in a loop over offsets to save memory.
    result_or = np.zeros(n_max - n_min + 1, dtype=np.uint8)
    
    for offset in range(24):
        idx = starts + 1 + offset
        result_or |= masks[idx]
    
    # Now count bits in each result_or value
    # Use a lookup table for popcount of 0..255
    popcount = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        popcount[i] = bin(i).count('1')
    
    counts = popcount[result_or]
    
    # Find the minimum count and the corresponding n
    min_count = np.min(counts)
    min_indices = np.where(counts == min_count)[0]
    best_n = n_min + min_indices[0]
    
    return int(best_n), int(min_count)

def _full_search(masks, deadline):
    """
    Do a full search over all n in [1000, 1000000] using the sliding window
    OR approach. Returns (best_n, min_distinct_count).
    """
    return _sliding_window_min_distinct(masks)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    # Precompute masks
    masks = _compute_masks()
    
    # Full search
    best_n, min_count = _full_search(masks, deadline)
    
    return best_n

# EVOLVE-BLOCK-END
