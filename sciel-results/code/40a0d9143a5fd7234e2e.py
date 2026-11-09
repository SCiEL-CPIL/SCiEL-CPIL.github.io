"""Rewritten program with linear-sieve LPF precomputation for O(1) witness checks."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _linear_lpf_sieve(n):
    """
    Linear sieve returning an array lpf where lpf[i] = least prime factor of i.
    lpf[i] = i if i is prime.
    Uses O(n) time and space.
    """
    lpf = list(range(n + 1))
    primes = []
    for i in range(2, n + 1):
        if lpf[i] == i:
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > n:
                break
            lpf[i * p] = p
    return lpf


def construct():
    """
    Finds integer witnesses m in (lo, hi) such that least_prime_factor(m) >= gap.
    Uses a precomputed LPF array for O(1) per-candidate checks.
    """
    deadline = time.time() + 55.0

    # Determine the maximum value we need LPF for
    max_prime = max(hi for lo, hi in P['gaps'])
    # We need LPF for all values up to max_prime (candidates are < max_prime)
    lpf = _linear_lpf_sieve(max_prime)

    result = [None] * len(P['gaps'])

    # For each gap, search for a witness
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break

        gap = hi - lo

        # Check candidates in order of decreasing LPF to find a valid witness faster.
        # But since gaps are small (at most 16), a simple scan is fine.
        # A witness m is valid iff lpf[m] >= gap.
        # Note: if m is prime, lpf[m] = m >= gap (since m > lo >= 2 and gap >= 2).
        # If m is composite, lpf[m] is its smallest prime factor.

        witness = None
        # Scan all integers strictly between lo and hi
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                witness = m
                break

        if witness is not None:
            result[i] = int(witness)

    return result
# EVOLVE-BLOCK-END
