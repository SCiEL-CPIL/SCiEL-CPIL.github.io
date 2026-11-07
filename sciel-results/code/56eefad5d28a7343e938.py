"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _good_residues(gap):
    """For a given gap, find residues mod primorial(primes < gap) that have no factor < gap."""
    if gap <= 2:
        return None  # All integers work for gap=2
    small_primes = [p for p in _sieve_primes(gap - 1) if p < gap]
    if not small_primes:
        return None
    # Compute primorial
    mod = 1
    for p in small_primes:
        mod *= p
    # Find good residues
    good = []
    for r in range(mod):
        ok = True
        for p in small_primes:
            if r % p == 0:
                ok = False
                break
        if ok:
            good.append(r)
    return (mod, good)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Precompute good residues for each unique gap
    gap_cache = {}
    for gap in set(hi - lo for lo, hi in P['gaps']):
        if gap <= 2:
            gap_cache[gap] = None
        else:
            small_primes = [p for p in _sieve_primes(gap - 1) if p < gap]
            if not small_primes:
                gap_cache[gap] = None
            else:
                mod = 1
                for p in small_primes:
                    mod *= p
                good = []
                for r in range(mod):
                    ok = True
                    for p in small_primes:
                        if r % p == 0:
                            ok = False
                            break
                    if ok:
                        good.append(r)
                gap_cache[gap] = (mod, good)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        cached = gap_cache[gap]
        
        if cached is None:
            # Gap <= 2 or no small primes: any number works
            # For gap=2, need odd number between two odd primes - there's exactly one
            if gap == 2:
                m = lo + 1
                if m < hi:
                    result[i] = m
            else:
                # Try all numbers
                for m in range(lo + 1, hi):
                    result[i] = m
                    break
        else:
            mod, good_residues = cached
            # Find candidates in the range that match good residues
            # m = lo + 1 + k, we need (lo + 1 + k) % mod in good_residues
            # Equivalently, k % mod in {r - (lo+1) % mod for r in good_residues}
            base = (lo + 1) % mod
            good_k_residues = [(r - base) % mod for r in good_residues]
            good_k_set = set(good_k_residues)
            
            found = False
            # Iterate k from 0 to gap-2 (since m = lo+1+k, and m < hi means k <= gap-2)
            for k in range(gap - 1):
                if k % mod in good_k_set:
                    m = lo + 1 + k
                    if m < hi:
                        result[i] = m
                        found = True
                        break
            
            if not found:
                # Fallback: brute force check
                for m in range(lo + 1, hi):
                    has_small = False
                    for p in _sieve_primes(gap - 1):
                        if p >= gap:
                            break
                        if m % p == 0:
                            has_small = True
                            break
                    if not has_small:
                        result[i] = m
                        found = True
                        break
        
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
