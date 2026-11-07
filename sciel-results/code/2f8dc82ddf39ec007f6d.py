"""LPF-sieve approach: precompute least prime factor for every integer up to
the maximum prime, then for each gap check whether any interior number has
LPF >= gap.  Select the number with the highest LPF as the witness.
"""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 75.0  # safety margin before the 90 s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

_max_val = max(hi for lo, hi in P['gaps'])

def _build_lpf(n):
    """Return an array lpf[0..n] where lpf[m] is the least prime factor of m,
    or m itself if m is prime. lpf[0]=0, lpf[1]=1."""
    lpf = np.arange(n + 1, dtype=np.int32)
    if n >= 0:
        lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, int(math.isqrt(n)) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

_lpf = _build_lpf(_max_val)

# ---------- construct ----------
def construct():
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - _START_TIME > _DEADLINE:
            break
        gap = hi - lo
        # Interior numbers: lo+1, lo+2, ..., hi-1
        seg = _lpf[lo + 1:hi]
        # Find indices where lpf >= gap
        valid_mask = seg >= gap
        if valid_mask.any():
            # Pick the one with the highest LPF (best witness)
            idx = int(np.argmax(seg[valid_mask]))
            # Map back to original index
            valid_indices = np.where(valid_mask)[0]
            best_local = valid_indices[idx]
            result[i] = int(lo + 1 + best_local)
    return result
