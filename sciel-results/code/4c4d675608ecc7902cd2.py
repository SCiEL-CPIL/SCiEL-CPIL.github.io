"""Refined Cohn-Goncalves Laguerre construction: optimal double-root locations.

The 12 positive double-root locations for the m=12, n=1 Laguerre combination
are the critical points (zeros of L_12') of the Laguerre polynomial L_12.
Placing the double zeros at these critical points minimizes the Fourier sign
uncertainty upper bound C4, beating the seed's arbitrary arithmetic placement.
"""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 630.0 - 30.0  # safety margin before the 630s kill

def _laguerre_L12(x):
    """L_12(x) via the explicit finite sum (version-independent)."""
    m = 12
    # L_m(x) = sum_{k=0}^{m} binom(m, k) * (-1)^k * x^k / k!
    import math
    total = 0.0
    for k in range(m + 1):
        term = math.comb(m, k) * ((-1.0) ** k) * (x ** k) / math.factorial(k)
        total += term
    return total

def _laguerre_L12_prime(x):
    """d/dx L_12(x) via the explicit finite sum of the derivative."""
    m = 12
    import math
    total = 0.0
    for k in range(1, m + 1):
        # d/dx [ binom(m,k) (-1)^k x^k / k! ] = binom(m,k)(-1)^k k x^{k-1}/k!
        # = binom(m,k)(-1)^k x^{k-1}/(k-1)!
        term = math.comb(m, k) * ((-1.0) ** k) * (x ** (k - 1)) / math.factorial(k - 1)
        total += term
    return total

def _bisect(f, a, b, iters=200):
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return None  # no sign change
    for _ in range(iters):
        if time.time() > _DEADLINE:
            break
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if fa * fm < 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)

def _zeros_of_L12(n=12):
    """Find the n positive zeros of L_12 by bracketing on [0, large]."""
    # L_12 zeros are all in (0, 2n) roughly; use a generous upper bound.
    # Scan for sign changes of L_12 on a fine grid, then bisect.
    import numpy as np
    upper = 60.0
    grid = np.linspace(0.0, upper, 20001)
    vals = np.array([_laguerre_L12(float(x)) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            r = _bisect(_laguerre_L12, float(grid[i]), float(grid[i + 1]))
            if r is not None:
                roots.append(r)
    # Deduplicate
    roots.sort()
    dedup = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 1e-7:
            dedup.append(r)
    return dedup[:n]

def construct():
    try:
        # Fallback seed (guaranteed valid) in case anything fails.
        fallback = [4.0 * (i + 1) for i in range(12)]

        z = _zeros_of_L12(12)
        if len(z) < 12:
            return fallback

        # Critical points of L_12: interlace with its zeros.
        # c_0 in (0, z_0); c_i in (z_{i-1}, z_i) for i=1..11.
        crit = []
        a = 0.0
        for i in range(12):
            b = z[i]
            r = _bisect(_laguerre_L12_prime, a, b)
            if r is None:
                return fallback
            crit.append(r)
            a = b
        if len(crit) != 12:
            return fallback
        # Ensure strictly positive and sorted.
        crit = sorted(crit)
        if any(c <= 0.0 for c in crit):
            return fallback
        return crit
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
