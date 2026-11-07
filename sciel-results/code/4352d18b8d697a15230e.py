"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_EDGES = P['edges']
_ESET = set()
_ADJ = [set() for _ in range(_N)]
for _u, _v in _EDGES:
    _ESET.add(frozenset((_u, _v)))
    _ADJ[_u].add(_v)
    _ADJ[_v].add(_u)
_ALLV = list(range(_N))

def _cross_edges(L, R):
    s = set()
    for a in L:
        for b in R:
            s.add(frozenset((a, b)))
    return s

def _validate(part):
    if not part:
        return False
    seen = set()
    for bic in part:
        if len(bic) != 2:
            return False
        L, R = bic
        if not L or not R:
            return False
        if any(a in R for a in L):
            return False
        for a in L:
            for b in R:
                e = frozenset((a, b))
                if e not in _ESET:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    if seen != _ESET:
        return False
    return True

def _find_max_biclique(remaining, adj, n, side_hint=None):
    L = list(remaining.keys())
    if not L:
        return None
    if side_hint is not None:
        L = [side_hint] + [x for x in L if x != side_hint]
    best = None
    for a in L:
        R0 = set(remaining[a])
        if not R0:
            continue
        Lcand = [a]
        for b in L:
            if b != a:
                inter = R0 & remaining[b]
                if len(inter) >= len(R0):
                    Lcand.append(b)
        for b in R0:
            Lc = set(Lcand)
            Lc.add(b)
            R0b = set()
            for x in Lc:
                R0b &= remaining[x]
            R0b.discard(b)
            if not R0b:
                continue
            Rf = [b]
            for c in R0b:
                inter = Lc & remaining[c]
                if len(inter) >= len(Lc):
                    Rf.append(c)
            Lset = set(Lcand) | {b}
            Rset = set(Rf)
            Lf = list(Lset)
            Rf = list(Rset)
            for x in Lf:
                for y in Rf:
                    if y not in remaining[x]:
                        break
                else:
                    continue
                break
            else:
                k = len(Lf) * len(Rf)
                if best is None or k > best[0]:
                    best = (k, Lf, Rf)
    if best is None:
        return None
    return best[1], best[2]

def _greedy_init():
    remaining = {v: set(_ADJ[v]) for v in _ALLV}
    part = []
    while remaining:
        L, R = _find_max_biclique(remaining, _ADJ, _N)
        if L is None:
            for v in list(remaining.keys()):
                nbrs = list(remaining[v])
                if not nbrs:
                    del remaining[v]
                    continue
                for w in nbrs:
                    remaining[v].discard(w)
                    remaining[w].discard(v)
                part.append([[v], [w]] if False else [[v], [w]])
            continue
        for a in L:
            for b in R:
                remaining[a].discard(b)
                remaining[b].discard(a)
        for v in list(remaining.keys()):
            if not remaining[v]:
                del remaining[v]
        part.append([list(L), list(R)])
    return part

def _try_move(part, v, target, deadline):
    L, R = part[target]
    if v in L or v in R:
        return False
    if any(v not in _ADJ[x] for x in L) or any(v not in _ADJ[x] for x in R):
        return False
    for i, (Li, Ri) in enumerate(part):
        if i == target:
            continue
        if v in Li:
            Li.remove(v)
            if not Li:
                part.pop(i)
                if target > i:
                    target -= 1
                R.append(v)
                return True
        elif v in Ri:
            Ri.remove(v)
            if not Ri:
                part.pop(i)
                if target > i:
                    target -= 1
                L.append(v)
                return True
    return False

def _try_merge(part, i, j, deadline):
    if i == j:
        return False
    Li, Ri = part[i]
    Lj, Rj = part[j]
    if not Li or not Ri or not Lj or not Rj:
        return False
    if any(x in Ri for x in Li) or any(x in Rj for x in Lj):
        return False
    for candL, candR in ((Li, Rj), (Ri, Lj)):
        if any(x not in _ADJ[y] for x in candL for y in candR):
            continue
        newL = list(candL)
        newR = list(candR)
        restL = [x for x in Li if x not in set(candL)]
        restR = [x for x in Ri if x not in set(candR)]
        restL2 = [x for x in Lj if x not in set(candL)]
        restR2 = [x for x in Rj if x not in set(candR)]
        if restL and restR:
            if any(x not in _ADJ[y] for x in restL for y in restR):
                continue
        if restL2 and restR2:
            if any(x not in _ADJ[y] for x in restL2 for y in restR2):
                continue
        newpart = []
        for k, (Lk, Rk) in enumerate(part):
            if k == i or k == j:
                continue
            newpart.append([list(Lk), list(Rk)])
        newpart.append([newL, newR])
        if restL and restR:
            newpart.append([restL, restR])
        if restL2 and restR2:
            newpart.append([restL2, restR2])
        if len(newpart) < len(part) and _validate(newpart):
            part[:] = newpart
            return True
    return False

def _local_search(part, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(part)
        for i in range(n):
            if time.time() > deadline:
                return part
            Li, Ri = part[i]
            for v in list(Li) + list(Ri):
                for j in range(n):
                    if j == i:
                        continue
                    if _try_move(part, v, j, deadline):
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if _try_merge(part, i, j, deadline):
                    improved = True
                    break
            if improved:
                break
    return part

def construct():
    deadline = time.time() + 80.0
    best = None
    best_count = float('inf')
    for trial in range(200):
        if time.time() > deadline:
            break
        part = _greedy_init()
        if not _validate(part):
            continue
        part = _local_search(part, deadline)
        if not _validate(part):
            continue
        c = len(part)
        if c < best_count:
            best_count = c
            best = part
        if best_count <= 4:
            break
    if best is None:
        best = _greedy_init()
    out = []
    for L, R in best:
        out.append([[int(x) for x in L], [int(y) for y in R]])
    return out
# EVOLVE-BLOCK-END
