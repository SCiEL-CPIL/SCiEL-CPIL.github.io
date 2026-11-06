import time
from fractions import Fraction

_TARGET = Fraction(5, 31)
_DEADLINE = 50.0

def _max_sum(start, k, B):
    if k <= 0:
        return Fraction(0)
    if start + k - 1 > B:
        return None
    s = Fraction(0)
    for i in range(k):
        s += Fraction(1, start + i)
    return s

def _min_sum(start, k, B):
    if k <= 0:
        return Fraction(0)
    if B - k + 1 < start:
        return None
    s = Fraction(0)
    for i in range(k):
        s += Fraction(1, B - i)
    return s

def construct():
    t0 = time.time()
    best = [None, None]
    B = 50000
    MAXT = 10

    def greedy():
        r = _TARGET
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0:
                break
            vals.append(d)
            r -= Fraction(1, d)
        return vals
    g = greedy()
    if g and sum(Fraction(1, d) for d in g) == _TARGET:
        best[0] = max(g)
        best[1] = g

    def dfs(target, last, depth, path):
        if time.time() - t0 > _DEADLINE:
            return
        if target == 0:
            m = path[-1]
            if best[0] is None or m < best[0]:
                best[0] = m
                best[1] = list(path)
            return
        if depth >= MAXT:
            return
        k = MAXT - depth
        mx = _max_sum(last + 1, k, B)
        if mx is None or target > mx:
            return
        mn = _min_sum(last + 1, k, B)
        if mn is not None and target < mn:
            return
        dmin = max(last + 1, (target.denominator + target.numerator - 1) // target.numerator)
        for d in range(dmin, B + 1):
            if time.time() - t0 > _DEADLINE:
                return
            nd = target - Fraction(1, d)
            if nd < 0:
                break
            if nd == 0:
                m = d
                if best[0] is None or m < best[0]:
                    best[0] = m
                    best[1] = list(path) + [d]
                continue
            k2 = k - 1
            mx2 = _max_sum(d + 1, k2, B)
            if mx2 is None or nd > mx2:
                # for larger d, mx2 decreases and nd increases -> break
                break
            mn2 = _min_sum(d + 1, k2, B)
            if mn2 is not None and nd < mn2:
                continue
            path.append(d)
            dfs(nd, d, depth + 1, path)
            path.pop()

    dfs(_TARGET, 6, 0, [])

    vals = best[1] if best[1] is not None else g
    assert 1 <= len(vals) <= 256
    assert all(1 <= d <= 10000000 for d in vals)
    assert len(set(vals)) == len(vals)
    assert sum(Fraction(1, d) for d in vals) == _TARGET
    return vals
