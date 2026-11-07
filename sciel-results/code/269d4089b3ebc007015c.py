"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Pre-compute primes up to a sufficient bound
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

PRIMES = sieve(200)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin out of 90s
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        candidates = list(range(lo + 1, hi))
        
        if not candidates:
            continue
        
        # For each candidate, compute its least prime factor
        # A candidate m is valid if lpf(m) >= gap
        # i.e., m is not divisible by any prime < gap
        
        # Get primes strictly less than gap
        relevant_primes = [p for p in PRIMES if p < gap]
        
        # Score each candidate: count how many relevant primes divide it
        # (fewer divisors by small primes = more likely valid)
        # Actually, we just need to find ANY valid one.
        # Strategy: check candidates in order of "primality-likeness"
        # For each candidate, check if divisible by any relevant prime.
        
        # Quick filter: eliminate candidates divisible by 2, 3, 5 first
        # Then check remaining primes
        
        witness = None
        
        # Sort candidates by number of small prime factors (ascending)
        # to find a valid witness faster
        def small_factor_count(m):
            count = 0
            for p in relevant_primes[:5]:  # just first few for sorting
                if m % p == 0:
                    count += 1
            return count
        
        # Try candidates in order of fewest small factors first
        # But also try from both ends
        sorted_candidates = sorted(candidates, key=small_factor_count)
        
        for m in sorted_candidates:
            valid = True
            for p in relevant_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                witness = m
                break
        
        result[i] = witness
    
    # Ensure all entries are proper Python ints or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result

# EVOLVE-BLOCK-END
