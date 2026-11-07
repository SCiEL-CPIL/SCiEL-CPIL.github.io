"""Rewritten program: modular arithmetic coverage for witness search in prime gaps."""
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

    # For each gap size, determine the forbidden primes (primes < gap)
    # and precompute the valid residues modulo the product of forbidden primes
    forbidden_by_gap = {}
    valid_residues_by_gap = {}
    for g in range(2, max_gap + 1):
        fp = [p for p in primes_up_to_max_gap if p < g]
        forbidden_by_gap[g] = fp
        if not fp:
            valid_residues_by_gap[g] = None  # all residues valid
        else:
            # Product of forbidden primes
            prod = 1
            for p in fp:
                prod *= p
            # Use CRT to find all residues mod prod that are coprime to all fp
            # A number m is valid iff gcd(m, prod) == 1
            # So valid residues are those r in [0, prod) with gcd(r, prod) == 1
            valid_residues = set()
            for r in range(prod):
                ok = True
                for p in fp:
                    if r % p == 0:
                        ok = False
                        break
                if ok:
                    valid_residues.add(r)
            valid_residues_by_gap[g] = valid_residues

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        fp = forbidden_by_gap[gap]
        vr = valid_residues_by_gap[gap]

        if vr is None:
            # All candidates are valid; pick lo+1
            result[i] = lo + 1
            continue

        prod = 1
        for p in fp:
            prod *= p

        found = False
        for m in range(lo + 1, hi):
            if m % prod in vr:
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
