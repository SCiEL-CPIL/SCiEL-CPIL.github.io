"""Structural rewrite: precompute a global least-prime-factor (LPF) array over the
entire range of interest, then for each gap select the integer m in (lo, hi) whose
LPF is maximised subject to LPF(m) >= gap.  This is a different modelling family from
the per-gap boolean 'valid' sieve: we compute one LPF table once, and the selection
criterion is 'best LPF' rather than 'first valid'.

The LPF of m is the smallest prime that divides m (or m itself if m is prime).
A witness m for gap g requires LPF(m) >= g, i.e. m has no prime factor < g.
Among all such m in (lo, hi), we pick the one with the largest LPF.  This maximises
the margin and may find witnesses in gaps where the 'first valid' heuristic fails
if the first valid number happens to have a small LPF (though any valid number
satisfies the constraint, so 'first valid' should also work; the real benefit is
that the LPF table gives us a single O(1) lookup per candidate, and we can use
NumPy argmax over the slice for speed).

Actually, the key structural difference: instead of building a separate boolean
array per gap size (which is O(num_gaps * max_val * num_small_primes)), we build
ONE LPF array of size max_val+1 in O(max_val log log max_val) via a modified sieve.
Then for each gap we just do a NumPy argmax on the LPF slice, masked by the
constraint LPF >= gap.  This is cleaner and potentially faster for many distinct
gap sizes.
"""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 75.0  # safety margin before the 90 s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

_max_val = max(hi for lo, hi in P['gaps'])
_min_val = min(lo for lo, hi in P['gaps'])

# ---------- Build global LPF array ----------
# lpf[n] = least prime factor of n, or n itself if n is prime.
# lpf[0] = 0, lpf[1] = 1 (neither is a valid witness anyway).
def _build_lpf(n):
    """Return a NumPy int32 array lpf of length n+1 where lpf[i] is the least
    prime factor of i (or i if i is prime)."""
    lpf = np.zeros(n + 1, dtype=np.int32)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, n + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, n + 1, i):
                if lpf[j] == 0:
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
        # Candidate integers: lo+1 .. hi-1
        if hi - lo <= 1:
            continue
        segment = _lpf[lo + 1:hi]  # LPF values for each candidate
        # We need LPF >= gap. Among those, pick the one with max LPF.
        # Create a masked array: set invalid entries to -1 so they don't win argmax.
        masked = np.where(segment >= gap, segment, -1)
        max_lpf = np.max(masked)
        if max_lpf < gap:
            # No valid witness in this gap
            result[i] = None
        else:
            # Find the index of the first occurrence of max_lpf in masked
            idx = np.argmax(masked)
            result[i] = int(lo + 1 + idx)
    return result
