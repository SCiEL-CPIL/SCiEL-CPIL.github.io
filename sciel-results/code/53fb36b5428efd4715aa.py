"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _build_least_prime_factor_table(max_n, primes):
    """Build an array lpf where lpf[x] is the least prime factor of x for x <= max_n.
    For primes, lpf[x] = x."""
    lpf = [0] * (max_n + 1)
    for p in primes:
        if p > max_n:
            break
        for multiple in range(p, max_n + 1, p):
            if lpf[multiple] == 0:
                lpf[multiple] = p
    # For any remaining 0 entries (shouldn't happen for composites, but for primes set to itself)
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            lpf[i] = i
    return lpf


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 90s limit

    # Determine max value we might need to check
    all_values = []
    for lo, hi in P['gaps']:
        all_values.extend(range(lo + 1, hi))
    if not all_values:
        return [None] * len(P['gaps'])

    max_val = max(all_values)
    # We need least prime factor up to max_val
    # But we only need primes up to isqrt(max_val) for the sieve of primes list
    max_prime_needed = isqrt(max_val)

    # Build primes up to isqrt(max_val)
    primes = _sieve_primes_up_to(max_prime_needed)

    # Build least prime factor table
    lpf = _build_least_prime_factor_table(max_val, primes)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Check if time is running out
        if time.time() > deadline:
            break

        # Search for a witness strictly between lo and hi
        # whose least prime factor is >= gap
        found = False
        # Try all candidates in the gap
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            lpf_val = lpf[m]
            if lpf_val >= gap:
                result[i] = m
                found = True
                break
        # If not found in the forward scan, we could also try backward,
        # but forward is fine since we just need one witness.

    # Ensure all values are Python int (not numpy)
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result

# EVOLVE-BLOCK-END
