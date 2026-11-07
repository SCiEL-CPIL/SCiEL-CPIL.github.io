"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            count = len(range(start, limit + 1, step))
            is_prime[start::step] = bytearray(count)
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    """Find witnesses for each prime gap."""
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # The maximum possible gap in our list
    max_gap = max(hi - lo for lo, hi in gaps)

    # We need primes up to max_gap - 1 to check divisibility
    # (A number's least prime factor must be < gap size to be invalid)
    forbidden_primes = _sieve_primes(max_gap - 1)

    # Precompute the specific forbidden primes for each distinct gap size
    # forbidden_primes[g] = tuple of primes p such that p < g
    gap_forbidden = {}
    for lo, hi in gaps:
        g = hi - lo
        if g not in gap_forbidden:
            # We only need to check divisibility by primes strictly less than the gap
            gap_forbidden[g] = tuple(p for p in forbidden_primes if p < g)

    # Process each gap
    # The search space is small (gaps <= 20, range <= 19 integers)
    # so this is extremely fast.
    for i, (lo, hi) in enumerate(gaps):
        g = hi - lo
        forbidden = gap_forbidden[g]
        
        if not forbidden:
            # No primes < g, so any integer works (since least prime factor of any m >= 2 > 0)
            # Wait, if g=2, forbidden is empty. Primes < 2 are none.
            # Is it possible that m has a prime factor < 2? No, smallest prime is 2.
            # So if g=2, any m works because lpf(m) >= 2 = g.
            # If g=1, m must be strictly between, impossible for consecutive primes.
            result[i] = lo + 1
            continue

        # Search for m in (lo, hi)
        # We can iterate in any order. Let's try all.
        # Optimization: iterate in order of increasing m? Or random?
        # Since we want to find ANY, and small numbers often have small factors,
        # maybe larger numbers are better?
        # Actually, let's just iterate through all candidates.
        
        found = False
        for m in range(lo + 1, hi):
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        # If not found, result[i] remains None

    # Ensure types are correct: Python int or None
    return [int(v) if v is not None else None for v in result]

# EVOLVE-BLOCK-END
