"""Rewritten program: MODEL-AND-OPTIMIZE over the 32 gaps, sieve-based search."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _primes_upto(n):
    """Return list of primes <= n (Sieve of Eratosthenes)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    import time
    deadline = time.time() + 80.0  # safety margin vs 90s wall-clock

    gaps = P['gaps']
    result = [None] * len(gaps)

    # Precompute all primes up to max gap - 1
    max_gap = max(hi - lo for lo, hi in gaps)
    all_primes = _primes_upto(max_gap - 1)

    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        # Need m in (lo, hi) with least prime factor >= gap,
        # i.e. m not divisible by any prime < gap.
        small = [p for p in all_primes if p < gap]
        if not small:
            # No small primes to exclude; any m in (lo,hi) works
            result[i] = int(lo + 1)
            continue

        # Sieve the range [lo+1, hi-1]
        n = hi - lo - 1  # number of candidates
        if n <= 0:
            continue
        ok = bytearray([1]) * n  # ok[j] = 1 means lo+1+j is a candidate

        for p in small:
            # Find first multiple of p in [lo+1, hi-1]
            first = ((lo + 1) + p - 1) // p * p
            if first >= hi:
                continue
            # Mark multiples of p
            for m in range(first, hi, p):
                idx = m - (lo + 1)
                if idx < n:
                    ok[idx] = 0

        # Find first unmarked candidate
        found = None
        for j in range(n):
            if ok[j]:
                found = int(lo + 1 + j)
                break
        if found is not None:
            result[i] = found
        if time.time() > deadline:
            break
    return result
# EVOLVE-BLOCK-END
