"""Rewritten program: large-modulus-first greedy + local-search polish for odd covering."""
import time
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
_DIVISORS = None
_CLASSES = None  # dict m -> list of bool arrays (length BASE) for each residue
_CLASSES_L = None  # list of (m, residue_arrays) for large->small ordering

def _ensure_precompute():
    global _DIVISORS, _CLASSES, _CLASSES_L
    if _DIVISORS is not None:
        return
    divs = []
    for d in range(2, BASE + 1):
        if BASE % d == 0:
            divs.append(d)
    # keep odd only (all are odd since BASE odd)
    _DIVISORS = divs
    classes = {}
    for m in divs:
        arrs = []
        for r in range(m):
            idx = np.arange(r, BASE, m)
            b = np.zeros(BASE, dtype=bool)
            b[idx] = True
            arrs.append(b)
        classes[m] = arrs
    _CLASSES = classes
    # large -> small ordering
    _CLASSES_L = sorted(classes.items(), key=lambda kv: kv[0], reverse=True)


def _score(covered):
    # number of uncovered residues
    return int(np.count_nonzero(~covered))


def _local_search(classes, covered, time_limit):
    """Try swapping residue of each chosen modulus; accept improvements."""
    cur_uncovered = _score(covered)
    # classes is dict m -> (residue, arr)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for m in sorted(classes.keys(), reverse=True):
            cur_r, _ = classes[m]
            arrs = _CLASSES[m]
            best_r = cur_r
            best_un = cur_uncovered
            # try other residues
            for r in range(m):
                if r == cur_r:
                    continue
                new_cov = covered & ~arrs[cur_r] | arrs[r]
                un = _score(new_cov)
                if un < best_un:
                    best_un = un
                    best_r = r
            if best_r != cur_r:
                covered = covered & ~arrs[cur_r] | arrs[best_r]
                classes[m] = (best_r, arrs[best_r])
                cur_uncovered = best_un
                improved = True
                if cur_uncovered == 0:
                    break
    return classes, covered, cur_uncovered


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s
    _ensure_precompute()

    covered = np.zeros(BASE, dtype=bool)
    chosen = {}  # m -> (r, arr)

    # Phase 1: large moduli first, greedy max marginal coverage
    for m, arrs in _CLASSES_L:
        if time.time() > deadline:
            break
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = int(np.count_nonzero(arrs[r] & ~covered))
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if best_gain == 0:
                    # no gain; still pick a residue (smallest)
                    break
        arr = arrs[best_r]
        covered = covered | arr
        chosen[m] = (best_r, arr)

    # Phase 2: local search (swap residues)
    if time.time() < deadline - 2.0:
        chosen, covered, _ = _local_search(chosen, covered, deadline - 1.0)

    # Build result
    classes = [[m, r] for m, (r, _) in chosen.items()]
    return classes
