"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt

    def least_prime_factor(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        d = 5
        while d * d <= n:
            if n % d == 0:
                return d
            if n % (d + 2) == 0:
                return d + 2
            d += 6
        return n

    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Search for m in (lo, hi) with least prime factor >= gap
        found = False
        for m in range(lo + 1, hi):
            lpf = least_prime_factor(m)
            if lpf >= gap:
                result[i] = m
                found = True
                break
        if not found:
            result[i] = None
    return result
# EVOLVE-BLOCK-END
