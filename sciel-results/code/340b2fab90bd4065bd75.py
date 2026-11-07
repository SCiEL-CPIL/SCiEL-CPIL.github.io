"""Model-and-optimize rewrite for the consecutive-prime-gap witness task."""
import time
import math

_START_TIME = time.time()
_DEADLINE = 80.0  # explicit safety margin before the 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]


# Precompute primes up to max gap + 1
_max_gap = max(hi - lo for lo, hi in P['gaps'])
_primes = _sieve(_max_gap + 1)


def _find_witness(lo, hi, gap, small_primes):
    """Find m in (lo, hi) with LPF(m) >= gap, or None if none exists.
    
    For these small gaps (max 28), a direct scalar scan is optimal.
    """
    if not small_primes:
        # gap <= 2: no primes to avoid, any interior point works
        return lo + 1
    # Scan all interior points; return first valid one
    for m in range(lo + 1, hi):
        for p in small_primes:
            if m % p == 0:
                break
        else:
            return m
    return None


def construct():
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _START_TIME > _DEADLINE:
            break
        gap = hi - lo
        small_primes = [p for p in _primes if p < gap]
        w = _find_witness(lo, hi, gap, small_primes)
        if w is not None:
            result[i] = int(w)
    return result
