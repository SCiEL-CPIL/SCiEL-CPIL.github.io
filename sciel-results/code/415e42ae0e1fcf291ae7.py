"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_gap = max(hi - lo for lo, hi in P['gaps'])

    # Sieve for primes up to max_gap
    if max_gap >= 2:
        sieve = [True] * (max_gap + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_gap**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, max_gap + 1, i):
                    sieve[j] = False
        primes_up_to_max_gap = [i for i in range(2, max_gap + 1) if sieve[i]]
    else:
        primes_up_to_max_gap = []

    # For each distinct gap size, precompute forbidden primes (primes < gap)
    # and also precompute a "validity" pattern: for each gap size, we can
    # precompute which residues mod LCM(forbidden_primes) are valid.
    # But LCM can be huge, so instead we precompute for each gap size a
    # function that quickly checks if a number is valid.
    
    # Better approach: for each gap size g, the forbidden primes are primes < g.
    # A number m is valid iff gcd(m, product_of_forbidden_primes) == 1
    # (i.e., m shares no prime factor with the product).
    
    # Precompute the product of forbidden primes for each gap size.
    # For small gaps, this product fits in Python int easily.
    forbidden_products = {}
    for g in range(2, max_gap + 1):
        prod = 1
        for p in primes_up_to_max_gap:
            if p < g:
                prod *= p
            else:
                break
        forbidden_products[g] = prod

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        prod = forbidden_products[gap]
        
        # Scan candidates from lo+1 to hi-1
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Check if m has any prime factor < gap
            # This is equivalent to gcd(m, prod) == 1
            import math
            if math.gcd(m, prod) == 1:
                result[i] = m
                found = True
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
