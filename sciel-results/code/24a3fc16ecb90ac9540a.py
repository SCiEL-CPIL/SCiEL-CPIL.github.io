"""Maximize pairwise-disjoint residue classes with distinct moduli 2..128."""
import math
import time
from itertools import combinations

_MODMIN, _MODMAX = 2, 128
_DEADLINE = 55.0  # safety margin before 60s limit


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _disjoint(r1, m1, r2, m2):
    """Two infinite classes a=r1 mod m1 and a=r2 mod m2 are disjoint iff
    gcd(m1,m2) does NOT divide (r1-r2)."""
    return _gcd(m1, m2) % ((r1 - r2) % _gcd(m1, m2) or _gcd(m1, m2)) != 0


def _solve(moduli):
    """Try to assign each modulus a residue in [0,m) so all classes are
    pairwise disjoint. Returns list of [residue, modulus] or None."""
    n = len(moduli)
    # Precompute gcd pairs
    g = [[_gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    # Start with a simple assignment: r_i = i % m_i
    res = [i % moduli[i] for i in range(n)]

    def check():
        for i in range(n):
            for j in range(i + 1, n):
                gg = g[i][j]
                if gg > 1 and (res[i] - res[j]) % gg == 0:
                    return False
        return True

    if check():
        return [[res[i], moduli[i]] for i in range(n)]

    # Local search: fix violated pairs by changing a residue
    best = list(res)
    for _ in range(20000):
        # find a violated pair
        violated = None
        for i in range(n):
            for j in range(i + 1, n):
                gg = g[i][j]
                if gg > 1 and (best[i] - best[j]) % gg == 0:
                    violated = (i, j)
                    break
            if violated:
                break
        if violated is None:
            return [[best[i], moduli[i]] for i in range(n)]
        i, j = violated
        # try changing best[i] to a value that breaks this pair and hopefully others
        gg = g[i][j]
        bad = (best[j] % gg)
        for newr in range(moduli[i]):
            if newr % gg != bad:
                ok = True
                for k in range(n):
                    if k == i:
                        continue
                    gik = g[i][k]
                    if gik > 1 and (newr - best[k]) % gik == 0:
                        ok = False
                        break
                if ok:
                    best[i] = newr
                    break
        else:
            # try changing j
            for newr in range(moduli[j]):
                if newr % gg != (best[i] % gg):
                    ok = True
                    for k in range(n):
                        if k == j:
                            continue
                        gjk = g[j][k]
                        if gjk > 1 and (best[j] - best[k]) % gjk == 0 and (newr - best[k]) % gjk == 0:
                            pass
                        # recompute
                    # simpler: just set and check whole
                    best[j] = newr
                    if check():
                        return [[best[i], moduli[i]] for i in range(n)]
                    # revert
                    best[j] = bad + (best[j] - bad)  # keep original
                    best[j] = (best[j])  # no-op, will re-loop
                    break
    return None


def _greedy_build(moduli):
    """Greedy: add moduli one at a time, pick a residue that is disjoint
    from all already-chosen classes."""
    chosen = []  # list of (residue, modulus)
    for m in moduli:
        placed = False
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                gg = _gcd(m, m2)
                if gg > 1 and (r - r2) % gg == 0:
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                placed = True
                break
        if not placed:
            # try all residues again with different order
            for r in range(m - 1, -1, -1):
                ok = True
                for (r2, m2) in chosen:
                    gg = _gcd(m, m2)
                    if gg > 1 and (r - r2) % gg == 0:
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    placed = True
                    break
    return [[r, m] for (r, m) in chosen]


def construct():
    t0 = time.time()
    moduli = list(range(_MODMIN, _MODMAX + 1))  # 2..128, 127 moduli

    # Attempt 1: solve all 127 simultaneously
    sol = _solve(moduli)
    if sol is not None and len(sol) == len(moduli):
        return sol

    # Attempt 2: greedy over all moduli
    if time.time() - t0 > _DEADLINE:
        return [[1, 2]]
    greedy = _greedy_build(moduli)
    if len(greedy) >= 127:
        return greedy

    # Attempt 3: greedy with different orderings, keep best
    best = greedy
    import random
    rng = random.Random(42)
    for _ in range(200):
        if time.time() - t0 > _DEADLINE - 5:
            break
        order = moduli[:]
        rng.shuffle(order)
        g = _greedy_build(order)
        if len(g) > len(best):
            best = g
        if len(best) == len(moduli):
            return best

    # Fallback: seed-style power-of-2 construction (guaranteed valid, 7 classes)
    if len(best) < 7:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return best
