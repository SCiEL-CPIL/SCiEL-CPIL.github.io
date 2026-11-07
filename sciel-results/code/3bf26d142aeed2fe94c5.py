"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _lpf_at_least(m, threshold):
    """Check if least prime factor of m is >= threshold.
    i.e., m is not divisible by any prime < threshold."""
    if threshold <= 2:
        return True  # lpf is always >= 2
    # Check divisibility by primes < threshold
    # For efficiency, only need to check primes up to min(sqrt(m), threshold-1)
    # But actually we need ALL primes < threshold, not just up to sqrt(m)
    # Wait: if m has a prime factor p < threshold, then lpf(m) <= p < threshold.
    # So we need to check if m is divisible by ANY prime < threshold.
    # We can check all primes < threshold, but for large thresholds this is expensive.
    # However, our gaps are small (< 30), so threshold < 30, primes < 30 are few.
    # Actually, we can optimize: if m < threshold*threshold, then any prime factor >= threshold
    # means m itself is prime or m = p*q where p,q >= threshold, so m >= threshold^2.
    # But this optimization isn't critical for small gaps.
    
    # Generate primes < threshold on the fly (cached)
    return all(m % p != 0 for p in _PRIMES_CACHE if p < threshold)

# Precompute primes up to a reasonable limit
_MAX_GAP = max(hi - lo for lo, hi in P['gaps'])
_PRIMES_CACHE = _sieve_primes(_MAX_GAP + 10)  # primes up to max_gap+10

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        witness = None
        # Search from lo+1 to hi-1
        # Try sequential first
        for m in range(lo + 1, hi):
            if _lpf_at_least(m, gap):
                witness = m
                break
        
        result[i] = witness
    
    return result

# EVOLVE-BLOCK-END
