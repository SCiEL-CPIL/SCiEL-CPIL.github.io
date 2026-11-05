"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_N_SMALL = len(_SMALL_PRIMES)

def _precompute_signatures(max_n):
    """
    For each number m in [1, max_n], compute a bitmask where bit k is set
    iff the k-th small prime divides m (i.e., has a non-zero exponent).
    This represents the 'exponent signature' of the small-prime part.
    """
    sig = np.zeros(max_n + 1, dtype=np.uint8)  # 9 bits needed, uint8 is fine
    for k, p in enumerate(_SMALL_PRIMES):
        bit = 1 << k
        for m in range(p, max_n + 1, p):
            sig[m] |= bit
    return sig

def _compute_window_scores(sig, window_size=24, max_n=1000000):
    """
    For each starting position n in [1, max_n - window_size + 1], compute the
    number of distinct signatures in the window [n, n+window_size-1].
    This is used as a proxy for the number of distinct small-prime factors.
    
    Note: The actual task asks for distinct factors (primes), not distinct signatures.
    However, two numbers with the same signature have the same set of small prime divisors.
    So the number of distinct signatures in the window is exactly the number of distinct
    sets of small prime divisors, which is NOT the same as the number of distinct primes.
    
    Wait, let me re-read the task: "MINIMIZE the number of distinct extracted factors."
    The extracted factor for n+i is the complete factor composed of primes strictly below 24.
    So for each n+i, we extract a number (the 23-smooth part). Then we count the number of
    DISTINCT such extracted factors across the 24 numbers.
    
    So it's the number of distinct VALUES of the 23-smooth parts, not the number of distinct
    sets of primes. Two numbers can have the same set of primes but different exponents,
    leading to different smooth parts.
    
    So my signature approach is wrong. I need to track the actual smooth part values.
    """
    pass

def _compute_smooth_parts(max_n):
    """
    For each m in [1, max_n], compute the largest divisor of m composed only of primes < 24.
    """
    smooth = np.arange(max_n + 1, dtype=np.int64)
    for p in _SMALL_PRIMES:
        for m in range(p, max_n + 1, p):
            while smooth[m] % p == 0:
                smooth[m] //= p
    # Wait, this is wrong. We want the LARGEST divisor composed of small primes,
    # which means we should REMOVE all large prime factors, not small ones.
    
    # Let me redo this.
    smooth = np.arange(max_n + 1, dtype=np.int64)
    for p in _SMALL_PRIMES:
        for m in range(p, max_n + 1, p):
            while smooth[m] % p == 0:
                smooth[m] //= p
    # Now smooth[m] is m with all small prime factors removed. That's the LARGE part.
    # The small part is m // smooth[m].
    small_part = np.arange(max_n + 1, dtype=np.int64)
    small_part[1:] = np.arange(1, max_n + 1, dtype=np.int64) // smooth[1:]
    return small_part

def _find_best_n(small_part, window_size=24, max_n=1000000, deadline=None):
    """
    Find n in [1000, max_n] such that the number of distinct values in
    small_part[n:n+window_size] is minimized.
    """
    best_count = window_size + 1
    best_n = 1000
    
    # Use a sliding window with a frequency map
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0
    
    # Initialize window at n=1000
    for i in range(1000, 1000 + window_size):
        val = small_part[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n - window_size + 1):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Remove element at n-1
        val_out = small_part[n - 1]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct -= 1
        
        # Add element at n+window_size-1
        val_in = small_part[n + window_size - 1]
        if freq[val_in] == 0:
            distinct += 1
        freq[val_in] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    max_n = 1000000
    window_size = 24
    
    # Precompute small parts
    small_part = _compute_smooth_parts(max_n)
    
    # Find best n
    best_n, best_count = _find_best_n(small_part, window_size, max_n, deadline)
    
    return best_n
# EVOLVE-BLOCK-END
