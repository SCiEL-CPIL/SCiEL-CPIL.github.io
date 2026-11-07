"""Approximate a ratio inside a divisor-sum fiber.

Strategy: precompute sigma(n) for all n <= 10^7 with a vectorized divisor-sum
sieve, group the numbers by their sigma value (the 'fiber'), and within each
fiber find the pair (a, b) minimizing |2a - 3b| with a sorted two-pointer
sweep.  The raw error is |a/b - 3/2| = |2a - 3b| / (2b), so minimizing the
numerator inside a fiber is exactly what we want.  The exact zero (a/b = 3/2,
i.e. 2a == 3b) is rejected because it cannot be scored.
"""
import time
import numpy as np

MAXN = 10_000_000
DEADLINE = 78.0          # safety margin below the 90 s wall-clock kill
T_START = time.time()


def _sigma_sieve(n):
    """Return an int64 array sig[0..n] with sig[k] = sum of divisors of k."""
    sig = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sig[d::d] += d
    return sig


def _best_pair(sorted_arr):
    """Return (a, b, diff) minimizing |2a - 3b| over the fiber, else None.

    a and b are chosen from the fiber (they may be equal).  The exact-zero
    case 2a == 3b is skipped because it yields an unscoreable raw value of 0.
    """
    a = np.asarray(sorted_arr, dtype=np.int64)
    m = a.size
    if m < 2:
        return None
    best = None
    i = 0
    j = 0
    while i < m and j < m:
        va = 2 * a[i]
        vb = 3 * a[j]
        d = va - vb
        if d > 0:
            j += 1
        elif d < 0:
            i += 1
        else:
            # exact 3/2 ratio -> raw value 0 -> invalid; move past it
            i += 1
            continue
        if best is None or abs(d) < best[2]:
            best = (int(a[i]), int(a[j]), int(abs(d)))
    return best


def construct():
    global T_START
    sig = _sigma_sieve(MAXN)

    # Group indices by sigma value; only fibers with >= 2 members matter.
    # np.unique with return_inverse gives a compact sigma-id per number.
    sigma_vals, inverse = np.unique(sig[1:], return_inverse=True)
    counts = np.bincount(inverse, minlength=sigma_vals.size)
    fiber_ids = np.nonzero(counts >= 2)[0]

    best = None          # (a, b, absdiff, b)
    for fid in fiber_ids:
        if time.time() > DEADLINE:
            break
        members = np.nonzero(inverse == fid)[0] + 1   # actual n values
        if members.size < 2:
            continue
        res = _best_pair(members)
        if res is None:
            continue
        aa, bb, dd = res
        if best is None or dd < best[2] or (dd == best[2] and bb > best[3]):
            best = (aa, bb, dd, bb)

    if best is None:
        # Fallback: a guaranteed valid sigma-fiber pair (sigma(11)=sigma(6)=12).
        return [11, 6]

    a, b = int(best[0]), int(best[1])
    # Final exact validity checks.
    if a <= 0 or b <= 0 or a > MAXN or b > MAXN:
        return [11, 6]
    if 2 * a == 3 * b:            # exact 3/2 -> raw 0 -> invalid
        return [11, 6]
    return [a, b]
