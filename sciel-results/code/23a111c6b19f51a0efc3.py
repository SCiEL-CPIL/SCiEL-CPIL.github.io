"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(limit):
    """Return bytearray where is_prime[i] is 1 iff i is prime, for 0<=i<=limit."""
    if limit < 2:
        return bytearray(limit + 1)
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = bytearray(len(range(i*i, limit+1, i)))
    return is_prime

def _find_witness(lo, hi, primes_below_g, deadline):
    """Find an integer strictly between lo and hi whose LPF >= hi-lo."""
    g = hi - lo
    lo_excl = lo + 1
    hi_excl = hi
    num_candidates = hi_excl - lo_excl
    if num_candidates <= 0:
        return None

    if not primes_below_g:
        return lo_excl

    # Sieve: alive[j] = 1 if lo_excl + j has no prime factor < g
    alive = bytearray(b'\x01') * num_candidates

    for p in primes_below_g:
        first_j = (p - lo_excl % p) % p
        if first_j >= num_candidates:
            continue
        # Mark multiples
        alive[first_j : num_candidates : p] = bytearray(len(range(first_j, num_candidates, p)))

    # Find first surviving candidate
    for j in range(num_candidates):
        if alive[j]:
            return lo_excl + j

    return None

def construct():
    deadline = time.time() + 85  # 5s safety margin within 90s limit
    n = len(P['gaps'])
    result = [None] * n

    # Precompute small primes up to max gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    sieve_small = _sieve(max_gap)
    all_small_primes = [p for p in range(2, max_gap + 1) if sieve_small[p]]

    # For each gap, find a witness using batched modular sieve
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        g = hi - lo

        # Primes strictly less than g (these would make LPF < g)
        primes_below_g = [p for p in all_small_primes if p < g]

        result[i] = _find_witness(lo, hi, primes_below_g, deadline)

    # Ensure types are exactly Python int or None
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
