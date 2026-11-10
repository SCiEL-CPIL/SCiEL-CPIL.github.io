"""Rewritten program: Hall's theorem with vectorized zeta transform for exhaustive scan."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 12
MAX_M = 1000000
MAX_L = 144
LCM_12 = 27720
FULL_MASK = (1 << N) - 1  # 0xFFF

# Precompute popcounts for all 4096 masks
_POPCOUNTS = np.array([bin(s).count('1') for s in range(1 << N)], dtype=np.int32)
# Precompute all non-empty subset masks (exclude 0)
_SUBSETS = np.arange(1, 1 << N, dtype=np.int32)
# Precompute complements for each subset
_COMPLEMENTS = FULL_MASK ^ _SUBSETS


def _has_matching_hall_np(m, L):
    """Return True iff (m, m+L] contains distinct b_i with i | b_i for i=1..N.
    Uses Hall's theorem with vectorized zeta transform."""
    # Build divisor masks: div_masks[j] has bit i set iff (i+1) divides (m+1+j)
    div_masks = np.zeros(L, dtype=np.uint32)
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        # First multiple of i in (m, m+L]
        start = m + 1
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            j = v - m - 1
            div_masks[j] |= bit
            v += i

    # Count frequency of each divisor mask value
    freq = np.bincount(div_masks, minlength=1 << N).astype(np.int32)

    # Zeta transform: H[t] = sum_{k subset of t} freq[k]
    # After transform, H[t] = number of elements whose divisor mask is a subset of t
    H = freq.copy()
    for i in range(N):
        bit = 1 << i
        # For all indices where bit i is set, add the value from index without bit i
        # Slicing trick: H[bit::2*bit] += H[0::2*bit]
        H[bit::2 * bit] += H[0::2 * bit]

    # For subset S (mask s), |N(S)| = L - H[complement of s]
    # Because H[comp(s)] counts elements whose mask is a subset of comp(s),
    # i.e., elements NOT divisible by any member of S.
    nS = L - H[_COMPLEMENTS]
    required = _POPCOUNTS[_SUBSETS]

    return bool(np.all(nS >= required))


def _compute_f_hall(m):
    """Compute f(N, m): least positive L with a matching in (m, m+L]."""
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching_hall_np(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _find_best(deadline):
    """Search for m maximizing f(N,m). Returns (best_m, best_f)."""
    t0 = time.time()
    best_m = 0
    best_f = 0

    # Exploit periodicity: f(12, m) depends only on m mod LCM_12
    # Scan all residues in [0, LCM_12)
    for m in range(0, LCM_12):
        if (time.time() - t0) > deadline - 2:
            break
        f = _compute_f_hall(m)
        if f > best_f:
            best_f = f
            best_m = m

    return best_m, best_f


def construct():
    deadline = 85.0  # safety margin before 90s kill
    t0 = time.time()
    best_m, best_f = _find_best(deadline)
    # Ensure type is Python int
    return int(best_m)
# EVOLVE-BLOCK-END
