import time
import numpy as np

N = 128
T_LO, T_HI = 128, 256  # inclusive target interval

def _peak_of(A):
    """Return (max multiplicity, coverage_ok) for A as a sorted list/set."""
    Aset = set(A)
    maxc = 0
    ok = True
    for t in range(T_LO, T_HI + 1):
        c = 0
        # unordered a<=b, a+b=t, a,b in A
        # a from max(0, t-128) to t//2, with a in A and t-a in A
        lo = max(0, t - 128)
        hi = t // 2
        for a in range(lo, hi + 1):
            if a in Aset and (t - a) in Aset:
                c += 1
        if c == 0:
            ok = False
        if c > maxc:
            maxc = c
    return maxc, ok

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill
    best = None  # (peak, A_list)
    best_peak = None

    # Parametric two-block family: A = [0,k] U [128-m,128]
    # Search over k, m in a reasonable range.
    for k in range(40, 90):
        if time.time() > DEADLINE:
            break
        for m in range(16, 60):
            if time.time() > DEADLINE:
                break
            A = list(range(0, k + 1)) + list(range(128 - m, 129))
            peak, ok = _peak_of(A)
            if ok:
                if best_peak is None or peak < best_peak:
                    best_peak = peak
                    best = list(A)

    # If we found something, return it; else fallback to seed.
    if best is not None:
        return best
    return list(range(64, 129))
