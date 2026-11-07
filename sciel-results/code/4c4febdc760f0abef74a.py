"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt

    # Precompute primes up to max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])

    # Sieve to get all primes up to max_gap
    sieve = [True] * (max_gap + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(max_gap) + 1):
        if sieve[i]:
            for j in range(i*i, max_gap + 1, i):
                sieve[j] = False
    all_primes = [i for i in range(2, max_gap + 1) if sieve[i]]

    # For each distinct gap size, precompute the set of forbidden primes (primes < gap)
    gap_to_forbidden = {}
    for lo, hi in P['gaps']:
        gap = hi - lo
        if gap not in gap_to_forbidden:
            gap_to_forbidden[gap] = [p for p in all_primes if p < gap]

    # For each distinct gap size, build a boolean array marking valid residues
    # modulo the product of forbidden primes. This allows O(1) lookup.
    gap_to_valid = {}
    for gap, forbidden in gap_to_forbidden.items():
        if not forbidden:
            # No forbidden primes, every integer is valid
            gap_to_valid[gap] = (None, None)
        else:
            # Compute the product of forbidden primes as the period
            period = 1
            for p in forbidden:
                period *= p
            # Build valid_residues array: True if residue r is not divisible by any forbidden prime
            valid_residues = [True] * period
            for p in forbidden:
                for r in range(p, period, p):
                    valid_residues[r] = False
            gap_to_valid[gap] = (period, valid_residues)

    # Now find witnesses for each pair
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        period, valid_residues = gap_to_valid[gap]

        if period is None:
            # No forbidden primes, any integer in (lo, hi) works
            result[i] = lo + 1
        else:
            # Find the first m in (lo, hi) such that valid_residues[m % period] is True
            for m in range(lo + 1, hi):
                if valid_residues[m % period]:
                    result[i] = m
                    break

    return result
# EVOLVE-BLOCK-END
