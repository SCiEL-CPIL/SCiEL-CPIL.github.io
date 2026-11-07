"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _least_prime_factor(n, primes_list):
    """Return the least prime factor of n, or n if n is prime."""
    if n < 2:
        return n
    for p in primes_list:
        if p * p > n:
            return n
        if n % p == 0:
            return p
    return n

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps
    
    # Precompute primes up to max gap size (max gap is 20)
    max_gap = max(hi - lo for lo, hi in gaps)
    # We need primes up to max_gap to check least factor >= gap
    # Actually, we need primes up to sqrt(max_candidate) to factor candidates
    # max candidate is ~10289, sqrt is ~101
    # But for the condition "least prime factor >= gap", we just need to check
    # divisibility by primes < gap. If none divide it, LPF >= gap.
    
    # Generate small primes up to 100 (sufficient for sqrt(10289))
    small_primes = []
    for x in range(2, 100):
        if _is_prime(x):
            small_primes.append(x)
    
    # For each gap, find a witness
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap_size
        # This means m is not divisible by any prime < gap_size
        # Primes to check: those in small_primes that are < gap_size
        check_primes = [p for p in small_primes if p < gap_size]
        
        found = False
        # Try candidates from lo+1 to hi-1
        # We can try both ends or just iterate
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Check if m has any prime factor < gap_size
            valid = True
            for p in check_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                found = True
                break
        
        if not found:
            result[i] = None
    
    # Ensure types are correct
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
