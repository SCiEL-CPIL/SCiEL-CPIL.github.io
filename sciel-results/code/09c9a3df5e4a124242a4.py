# EVOLVE-BLOCK-START
import time
from math import gcd

_START = time.time()
_DEADLINE = 90.0 - 5.0  # safety margin before the 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_valid(moduli, residues):
    n = len(moduli)
    for i in range(n):
        ri = residues[i]
        mi = moduli[i]
        for j in range(i + 1, n):
            g = gcd(mi, moduli[j])
            if (ri - residues[j]) % g == 0:
                return False
    return True

def _solve_even_moduli(count):
    """Try to assign residues to the first `count` even moduli (2,4,...,2*count)."""
    moduli = [2 * (k + 1) for k in range(count)]  # 2,4,...,2*count
    n = count
    g = [[gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    # Variable ordering: most constrained first (smallest modulus => fewest choices,
    # and tends to have small gcds with many others).
    order = sorted(range(n), key=lambda i: moduli[i])
    assigned = [-1] * n

    def backtrack(pos):
        if _time_left() < 0.3:
            return None
        if pos == n:
            return assigned[:]
        idx = order[pos]
        m = moduli[idx]
        for r in range(m):
            ok = True
            for q in range(pos):
                j = order[q]
                if (r - assigned[j]) % g[idx][j] == 0:
                    ok = False
                    break
            if ok:
                assigned[idx] = r
                res = backtrack(pos + 1)
                if res is not None:
                    return res
        assigned[idx] = -1
        return None

    return backtrack(0)

def _fallback():
    # Guaranteed-valid small set under the CRT disjointness criterion
    # gcd(m_i, m_j) must NOT divide (r_i - r_j) for every pair.
    moduli = [6, 10, 15, 21, 35, 14]
    residues = [0, 1, 2, 3, 4, 5]
    if _is_valid(moduli, residues):
        return [[residues[i], moduli[i]] for i in range(len(moduli))]
    # Ultimate safe fallback (3-element triangle, verified valid)
    return [[0, 6], [1, 10], [2, 15]]

def construct():
    best = None
    # Attempt progressively smaller even-modulus sets until one solves in time.
    for count in (64, 48, 32, 24, 16, 12, 10, 8):
        if _time_left() < 2.0:
            break
        try:
            res = _solve_even_moduli(count)
            if res is not None:
                moduli = [2 * (k + 1) for k in range(count)]
                if _is_valid(moduli, res):
                    best = [[res[i], moduli[i]] for i in range(count)]
                    break
        except Exception:
            continue
    if best is not None:
        return best
    return _fallback()
# EVOLVE-BLOCK-END
