# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Determine the range we need to cover
    min_val = min(lo for lo, hi in gaps)
    max_val = max(hi for lo, hi in gaps)
    
    # We need LPF for numbers in (min_val, max_val)
    # Actually, we need LPF for numbers in each gap (lo, hi)
    # The smallest number we care about is min(lo+1 for lo,hi in gaps)
    # The largest is max(hi-1 for lo,hi in gaps)
    
    # For LPF computation, we need primes up to sqrt(max_val)
    sqrt_max = isqrt(max_val) + 1
    small_primes = _sieve_primes(sqrt_max)
    
    # Segmented sieve to compute LPF for all numbers in [min_val+1, max_val-1]
    # Actually, let's compute LPF for a range that covers all gaps
    range_start = min(lo + 1 for lo, hi in gaps)
    range_end = max(hi - 1 for lo, hi in gaps)
    
    # Create an array to store LPF for each number in [range_start, range_end]
    # lpf[m - range_start] = least prime factor of m
    # Initialize to 0, meaning "not yet determined"
    # For primes, LPF = the number itself
    
    num_entries = range_end - range_start + 1
    lpf = [0] * num_entries  # 0 means not set yet
    
    # For each prime p, mark multiples in our range
    # A number m has LPF = p if p is the smallest prime that divides m
    # So we process primes in increasing order, and for each prime p,
    # mark all multiples of p in our range that haven't been marked yet
    
    for p in small_primes:
        if p * p > max_val:
            break
        
        # Find the first multiple of p in [range_start, range_end]
        first = ((range_start + p - 1) // p) * p
        if first > range_end:
            continue
        
        # Mark all multiples of p in the range
        for m in range(first, range_end + 1, p):
            idx = m - range_start
            if lpf[idx] == 0:
                lpf[idx] = p
    
    # For numbers that are still 0, they are either prime or have no prime factor <= sqrt(max_val)
    # In our range, if lpf[idx] == 0, then the number is prime (since all its prime factors > sqrt(max_val)
    # would mean the number itself is > max_val, contradiction)
    # Actually, if a number m has no prime factor <= sqrt(m), then m is prime
    # Since sqrt(m) <= sqrt(max_val) for all m in our range, if lpf[idx] == 0, m is prime
    # So LPF(m) = m for those
    
    for idx in range(num_entries):
        if lpf[idx] == 0:
            lpf[idx] = range_start + idx  # The number itself is prime
    
    # Now for each gap, scan the precomputed LPF array to find a witness
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        
        gap_size = hi - lo
        
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            idx = m - range_start
            if lpf[idx] >= gap_size:
                result[i] = m
                break
    
    # Ensure all entries are Python int or None
    for i in range(n):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END
