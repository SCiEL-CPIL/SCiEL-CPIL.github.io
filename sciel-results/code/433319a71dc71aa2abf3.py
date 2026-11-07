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

def _least_prime_factor(m, small_primes):
    """Return least prime factor of m. If m is prime, return m."""
    for p in small_primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    # m has no small prime factor; check if it's prime or has larger factors
    # For our purposes, if no small prime divides it and p*p > m, it's prime
    # But we need to be careful: m might have a factor between max(small_primes) and sqrt(m)
    # However, since we only need least factor >= gap, and gaps are <= 32,
    # if m has no factor < 32, then least factor >= 32 >= gap for all our gaps
    # Actually we need least factor >= gap, so we need to check up to gap-1
    return m  # This is safe if small_primes covers up to at least max_gap

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    # Precompute primes up to 100 (covers all possible gaps which are <= 32)
    # We need primes up to max_gap to check if least factor >= gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need to check divisibility by all primes < max_gap
    # If m is not divisible by any prime < max_gap, then least factor >= max_gap >= gap
    small_primes = _sieve_primes(max_gap)  # primes up to max_gap
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Find candidates strictly between lo and hi
        # We want least prime factor >= gap
        # Strategy: try numbers, check if any prime < gap divides them
        found = False
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            has_small_factor = False
            for p in small_primes:
                if p >= gap:
                    break
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                result[i] = m
                found = True
                break
        # If not found, try more thoroughly (but for these gaps, should find)
        if not found:
            # More careful: check all integers up to isqrt(m) for factors
            for m in range(lo + 1, hi):
                # Find least prime factor
                lpf = m
                for d in range(2, isqrt(m) + 1):
                    if m % d == 0:
                        lpf = d
                        break
                if lpf >= gap:
                    result[i] = m
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
