"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s wall-clock kill

def _triples(n):
    return [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]

def _mod3_colors(n):
    return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in _triples(n)]

def _validate(n, colors):
    """Check no red K4^3 and no blue K5^3."""
    if len(colors) != n * (n - 1) * (n - 2) // 6:
        return False
    # index lookup
    idx = {}
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                idx[(a, b, c)] = k
                k += 1
    red = set(i for i, v in enumerate(colors) if v == 0)
    # red K4: all 4 triples red
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (idx[(a, b, c)] in red and idx[(a, b, d)] in red and
                idx[(a, c, d)] in red and idx[(b, c, d)] in red):
            return False
    # blue K5: all 10 triples blue
    blue = set(i for i, v in enumerate(colors) if v == 1)
    for quint in combinations(range(n), 5):
        ok = True
        for i in range(5):
            for j in range(i + 1, 5):
                for k2 in range(j + 1, 5):
                    if idx[(quint[i], quint[j], quint[k2])] not in blue:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return False
    return True

def _local_search(n, colors, budget):
    """Flip random triples, keep if still valid. Returns improved colors or None."""
    cur = list(colors)
    if not _validate(n, cur):
        return None
    triples = _triples(n)
    idx = {}
    k = 0
    for t in triples:
        idx[t] = k
        k += 1
    # precompute, for each triple, the red-K4 partner triples (3 others sharing 3 verts)
    # A red K4 is {a,b,c,d} -> triples (a,b,c),(a,b,d),(a,c,d),(b,c,d).
    # For triple (a,b,c), a red K4 is completed by a 4th vertex d with the other 3 red.
    red = [v == 0 for v in cur]
    for _ in range(200000):
        if time.time() > budget:
            break
        i = random.randrange(len(cur))
        a, b, c = triples[i]
        # try flip
        red[i] = not red[i]
        # check: does this create a red K4 or blue K5?
        bad = False
        if red[i]:  # became red; check red K4 with any d
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                if red[idx[(a, b, d)]] and red[idx[(a, c, d)]] and red[idx[(b, c, d)]]:
                    bad = True
                    break
        else:  # became blue; check blue K5 with any pair d,e
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                for e in range(d + 1, n):
                    if e == a or e == b or e == c:
                        continue
                    if (red[idx[(a, b, d)]] == False and red[idx[(a, c, d)]] == False and
                            red[idx[(b, c, d)]] == False and red[idx[(a, b, e)]] == False and
                            red[idx[(a, c, e)]] == False and red[idx[(b, c, e)]] == False and
                            red[idx[(a, d, e)]] == False and red[idx[(b, d, e)]] == False and
                            red[idx[(c, d, e)]] == False):
                        bad = True
                        break
                if bad:
                    break
        if bad:
            red[i] = not red[i]  # revert
        else:
            cur[i] = 0 if red[i] else 1
    return cur

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 135.0

    best_n = 0
    best_colors = None

    # 1) seed baseline
    seed_n = 9
    seed_colors = _mod3_colors(seed_n)
    if _validate(seed_n, seed_colors):
        best_n, best_colors = seed_n, seed_colors

    # 2) search for valid modular-construction n (sum mod m)
    for n in range(40, 9, -1):
        if time.time() > _DEADLINE - 20:
            break
        for m in range(3, n + 1):
            if time.time() > _DEADLINE - 20:
                break
            colors = [0 if (a + b + c) % m == 0 else 1 for a, b, c in _triples(n)]
            if _validate(n, colors):
                if n > best_n:
                    best_n, best_colors = n, colors
                break  # found valid for this n, move to larger? we go downward, so stop
        if best_n >= 34:
            break

    # 3) local-search refinement to push n higher
    if best_n > 0:
        improved = _local_search(best_n, best_colors, _DEADLINE - 10)
        if improved is not None and _validate(best_n, improved):
            best_colors = improved

    # 4) try to grow: start from best, add vertices via local search on n+1
    # (best effort, time-bounded)
    if time.time() < _DEADLINE - 15 and best_n < 40:
        # attempt n+1 from a random base
        n2 = best_n + 1
        base = _mod3_colors(n2)
        improved2 = _local_search(n2, base, _DEADLINE - 10)
        if improved2 is not None and _validate(n2, improved2):
            best_n, best_colors = n2, improved2

    if best_n == 0:
        return {'n': 9, 'colors': seed_colors}
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
