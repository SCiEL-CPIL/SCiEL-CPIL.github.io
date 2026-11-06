"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_PRIMES = len(_SMALL_PRIMES)
_MASK_SIZE = 1 << _NUM_PRIMES  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute bitmask for each number in [lower, max_val]
    # Bit j is set if the j-th prime divides the number
    masks = np.zeros(nvals, dtype=np.int32)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for j, p in enumerate(_SMALL_PRIMES):
        bit = 1 << j
        # A number has bit j set if it's divisible by p
        mask = (arr % p == 0)
        masks[mask] |= bit

    # Use sliding window with per-bit frequency counts for O(1) update per step
    # Precompute for each possible mask value (0..511), which bits are set
    mask_bits = []
    for m in range(_MASK_SIZE):
        bits = []
        for j in range(_NUM_PRIMES):
            if m & (1 << j):
                bits.append(j)
        mask_bits.append(bits)
    
    best_n = lower
    best_count = _NUM_PRIMES + 1  # max possible is 9
    
    # Initialize per-bit counts for window masks[1:25]
    bit_counts = [0] * _NUM_PRIMES
    for i in range(1, 25):
        m = masks[i]
        for j in mask_bits[m]:
            bit_counts[j] += 1
    
    # Count distinct bits
    distinct = sum(1 for j in range(_NUM_PRIMES) if bit_counts[j] > 0)
    best_count = distinct
    if best_count == 1:
        return best_n

    # Slide the window: offset from 1 to (upper - lower)
    # Window for n=lower+offset is masks[offset+1 : offset+25]
    n_steps = upper - lower
    check_interval = 50000
    masks_list = masks.tolist()  # Convert to list for faster indexing in Python loop
    
    for offset in range(1, n_steps + 1):
        if offset % check_interval == 0:
            if time.time() > deadline:
                break
        
        # Remove masks[offset] (leaving the window)
        val_out = masks_list[offset]
        for j in mask_bits[val_out]:
            bit_counts[j] -= 1
        
        # Add masks[offset + 24] (entering the window)
        val_in = masks_list[offset + 24]
        for j in mask_bits[val_in]:
            bit_counts[j] += 1
        
        # Count distinct bits
        distinct = sum(1 for j in range(_NUM_PRIMES) if bit_counts[j] > 0)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
# EVOLVE-BLOCK-END
