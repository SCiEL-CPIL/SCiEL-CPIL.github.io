import time
import math

_START_TIME = time.time()
_DEADLINE = 55.0  # Safety margin before 60s limit

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _build_lpf_sieve(max_val):
    """
    Build an array where lpf[i] is the least prime factor of i.
    lpf[0] and lpf[1] are 0. For primes p, lpf[p] = p.
    """
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    return lpf

# Precompute LPF up to the maximum prime in the pairs
_max_prime = max(hi for lo, hi in P['gaps'])
_lpf = _build_lpf_sieve(_max_prime)

def construct():
    """
    Returns a list of 32 entries, each an int or None.
    Entry i is a witness for the i-th prime pair, or None if no witness exists.
    """
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        # Check deadline
        if time.time() - _START_TIME > _DEADLINE:
            break
            
        gap = hi - lo
        
        # Search for witness m in (lo, hi) such that LPF(m) >= gap
        # Note: LPF(m) >= gap implies m has no prime factors < gap.
        # This is equivalent to m being coprime to the product of all primes < gap.
        # However, using the precomputed LPF array is O(1) per candidate.
        
        found = None
        for m in range(lo + 1, hi):
            if _lpf[m] >= gap:
                found = m
                break
        
        if found is not None:
            result[i] = int(found)
            
    return result
