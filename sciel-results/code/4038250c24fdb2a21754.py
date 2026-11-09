"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_lpf_up_to(n):
    """Return array lpf[0..n] where lpf[x] is the least prime factor of x (or x if prime)."""
    lpf = np.zeros(n + 1, dtype=np.int32)
    # Sieve: for each prime p, mark multiples that don't yet have an LPF
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    # For primes (and 1), lpf is still 0; set lpf[x] = x for x >= 2
    mask = (lpf == 0)
    idx = np.where(mask)[0]
    idx = idx[idx >= 2]
    lpf[idx] = idx
    return lpf


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin before 90s limit

    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    # Determine the maximum value we need LPF for
    max_val = max(hi for lo, hi in gaps)

    # Build LPF table in a single efficient pass
    lpf = _sieve_lpf_up_to(max_val)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break

        gap_size = hi - lo
        # Candidates are lo+1, lo+2, ..., hi-1
        # We need LPF(m) >= gap_size for some m in (lo, hi)
        
        # Extract LPF values for the candidate range [lo+1, hi-1]
        lpf_vals = lpf[lo + 1:hi]
        
        if len(lpf_vals) == 0:
            continue
        
        # Find indices where lpf >= gap_size
        valid_mask = lpf_vals >= gap_size
        if np.any(valid_mask):
            # Take the first valid candidate
            idx = np.argmax(valid_mask)
            result[i] = int(lo + 1 + idx)

    # Ensure all entries are Python int or None
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))

    return final_result
# EVOLVE-BLOCK-END
