import time
import numpy as np

# Primes strictly below 24
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_PRIMES = len(_PRIMES)

def _compute_signatures(max_n):
    """
    For each integer m in [0, max_n], compute a bitmask where bit k is set
    if prime _PRIMES[k] divides m (i.e., exponent >= 1).
    
    We use a sieve-like approach: for each prime p, mark all multiples of p.
    """
    sig = np.zeros(max_n + 1, dtype=np.int32)
    for k, p in enumerate(_PRIMES):
        mask = 1 << k
        # Mark all multiples of p
        for multiple in range(p, max_n + 1, p):
            sig[multiple] |= mask
    return sig

def _find_best_window(sig, window_size=24, start=1000, end=1000000, deadline=None):
    """
    Find the window [n, n+1, ..., n+23] (i.e., n+1 to n+24) that minimizes
    the number of distinct signatures.
    
    sig[m] gives the signature for m.
    We want n in [1000, 1000000] such that the window is n+1, n+2, ..., n+24.
    So we look at sig[n+1], sig[n+2], ..., sig[n+24].
    
    Using a sliding window with a frequency counter over signatures.
    """
    # The window covers indices n+1 to n+24 in sig
    # n ranges from 1000 to 1000000
    # So sig indices range from 1001 to 1000024
    
    min_score = window_size + 1  # worst case is all distinct
    best_n = start
    best_score = min_score
    
    # We'll use a frequency array. Signatures are 9-bit, so max 512 distinct values.
    freq = np.zeros(512, dtype=np.int32)
    unique_count = 0
    
    # Initialize first window: n = start, window is sig[start+1] to sig[start+24]
    for i in range(start + 1, start + 1 + window_size):
        s = sig[i]
        if freq[s] == 0:
            unique_count += 1
        freq[s] += 1
    
    if unique_count < best_score:
        best_score = unique_count
        best_n = start
    
    # Slide the window
    # Next window: n = start+1, window is sig[start+2] to sig[start+25]
    # Remove sig[start+1], add sig[start+1+window_size]
    for n in range(start + 1, end + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        # Remove sig[n] (the leftmost element leaving the window)
        s_remove = sig[n]
        freq[s_remove] -= 1
        if freq[s_remove] == 0:
            unique_count -= 1
        
        # Add sig[n + window_size] (the rightmost element entering the window)
        s_add = sig[n + window_size]
        if freq[s_add] == 0:
            unique_count += 1
        freq[s_add] += 1
        
        if unique_count < best_score:
            best_score = unique_count
            best_n = n
            if best_score <= 2:  # Can't do better than a small number
                # Continue searching but we can break early if we find very good
                pass
    
    return best_n, best_score

def construct():
    deadline = time.time() + 80  # 80 seconds, leave 10s safety margin
    
    # Compute signatures up to max needed: 1000000 + 24
    max_needed = 1000000 + 24
    sig = _compute_signatures(max_needed)
    
    if time.time() > deadline:
        return 1000
    
    # Find best window
    best_n, best_score = _find_best_window(sig, window_size=24, start=1000, end=1000000, deadline=deadline)
    
    return best_n

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
