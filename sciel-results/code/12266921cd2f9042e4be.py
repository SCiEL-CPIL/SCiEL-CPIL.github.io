import time
import sys

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_PRIMES = len(_SMALL_PRIMES)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute bitmask for each number
    # mask[i] corresponds to number (lower + i)
    # bit j is 1 if _SMALL_PRIMES[j] divides (lower + i)
    masks = [0] * nvals
    
    for j, p in enumerate(_SMALL_PRIMES):
        bit = 1 << j
        # Find first multiple of p >= lower
        first_mult = ((lower + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - lower
            masks[idx] |= bit

    # Sliding window
    # prime_counts[j] = number of elements in current window divisible by _SMALL_PRIMES[j]
    prime_counts = [0] * _NUM_PRIMES
    
    best_n = lower
    best_count = float('inf')
    
    window_size = 24
    
    # Initialize window for n=lower (indices 1 to 24 in masks array, since n+1..n+24)
    # Note: masks[0] is for 'lower'. The window is n+1 to n+24.
    # So for n=lower, we use masks[1] to masks[24].
    for i in range(1, 25):
        m = masks[i]
        for j in range(_NUM_PRIMES):
            if (m >> j) & 1:
                prime_counts[j] += 1
                
    distinct = sum(1 for c in prime_counts if c > 0)
    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide window
    # For n from lower+1 to upper:
    # Previous window: indices offset .. offset+23 (relative to lower)
    #   where offset = n - lower
    #   Wait, for n=lower, offset=0, window indices 1..24.
    #   For n=lower+1, offset=1, window indices 2..25.
    #   So we remove index `offset` and add index `offset+24`.
    
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break
            
        # Remove index `offset` (which is n)
        # The element leaving the window is n = lower + offset
        # In the previous step (n=lower+offset-1), the window was indices offset..offset+23?
        # Let's trace:
        # n=lower (offset 0): window masks[1..24].
        # n=lower+1 (offset 1): window masks[2..25].
        #   Remove masks[1], Add masks[25].
        #   General step for offset k (n = lower + k):
        #   Window is masks[k+1 .. k+24].
        #   Previous window (offset k-1) was masks[k .. k+23].
        #   So we remove masks[k] and add masks[k+24].
        
        m_out = masks[offset]
        for j in range(_NUM_PRIMES):
            if (m_out >> j) & 1:
                prime_counts[j] -= 1
                
        m_in = masks[offset + 24]
        for j in range(_NUM_PRIMES):
            if (m_in >> j) & 1:
                prime_counts[j] += 1
                
        distinct = sum(1 for c in prime_counts if c > 0)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 2:
                # Theoretical minimum is 2 (primes 2 and 3)
                # We can break early if we are confident, but let's keep scanning to be safe or just let it run as it's fast.
                # Actually, 2 is the min possible. If we find 2, we can return immediately to save time?
                # Yes, 2 is the global minimum.
                return best_n

    return best_n
