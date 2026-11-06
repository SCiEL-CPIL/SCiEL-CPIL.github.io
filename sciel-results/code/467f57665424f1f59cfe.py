import time
import random

N = 40
TARGET = 37

# Precompute coverage masks
_CHORD_MASKS = {}
for k in range(2, 39):
    mask = 0
    l1 = k + 1
    l2 = N - k + 1
    if 3 <= l1 <= 39:
        mask |= (1 << (l1 - 3))
    if 3 <= l2 <= 39:
        mask |= (1 << (l2 - 3))
    _CHORD_MASKS[k] = mask

_PAIR_MASK = {}
for u in range(2, 39):
    for v in range(u + 1, 39):
        l = v - u + 2
        _PAIR_MASK[(u, v)] = (1 << (l - 3)) if 3 <= l <= 39 else 0

_FULL_MASK = (1 << TARGET) - 1
_CANDIDATES = list(range(2, 39))

def _coverage(S):
    mask = 0
    sl = sorted(S)
    for k in sl:
        mask |= _CHORD_MASKS[k]
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            mask |= _PAIR_MASK[(sl[i], sl[j])]
    return mask

def _build_solution(S):
    S = sorted(S)
    edges = [(i, (i + 1) % N) for i in range(N)]
    for k in S:
        edges.append((0, k))
    edge_set = set((min(e), max(e)) for e in edges)
    length_map = {N: ('rim',)}
    for k in S:
        l1, l2 = k + 1, N - k + 1
        if 3 <= l1 <= N and l1 not in length_map:
            length_map[l1] = ('chord_short', k)
        if 3 <= l2 <= N and l2 not in length_map:
            length_map[l2] = ('chord_long', k)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            u, v = S[i], S[j]
            l = v - u + 2
            if 3 <= l <= N and l not in length_map:
                length_map[l] = ('chord_pair', u, v)
    if len(length_map) < N - 2:
        return None
    cycles = []
    for L in range(3, N + 1):
        if L not in length_map:
            return None
        t = length_map[L]
        if t[0] == 'rim':
            cycles.append(list(range(N)))
        elif t[0] == 'chord_short':
            cycles.append([0] + list(range(1, t[1] + 1)))
        elif t[0] == 'chord_long':
            cycles.append([0] + list(range(N - 1, t[1] - 1, -1)))
        else:
            u, v = t[1], t[2]
            cycles.append([0, u] + list(range(u + 1, v + 1)))
    for c in cycles:
        if len(set(c)) != len(c):
            return None
        for i in range(len(c)):
            a, b = c[i], c[(i + 1) % len(c)]
            if (min(a, b), max(a, b)) not in edge_set:
                return None
    return {'edges': edges, 'cycles': cycles}

def construct():
    start = time.time()
    deadline = start + 55.0

    for m in (8, 9, 10, 11):
        if time.time() > deadline:
            break
        # Greedy initialization
        S = set()
        for _ in range(m):
            bk, bm = None, 0
            for k in _CANDIDATES:
                if k in S:
                    continue
                ns = S | {k}
                mk = _coverage(ns)
                if mk > bm:
                    bm, bk = mk, k
            S.add(bk)
        if _coverage(S) == _FULL_MASK:
            sol = _build_solution(S)
            if sol:
                return sol
        # Local search with restarts
        for restart in range(300):
            if time.time() > deadline:
                break
            if restart == 0:
                S = set(S)
            else:
                S = set(random.sample(_CANDIDATES, m))
            best = _coverage(S)
            for _ in range(20000):
                if time.time() > deadline:
                    break
                if best == _FULL_MASK:
                    sol = _build_solution(S)
                    if sol:
                        return sol
                ins = list(S)
                outs = [k for k in _CANDIDATES if k not in S]
                if not ins or not outs:
                    break
                u, v = random.choice(ins), random.choice(outs)
                S.discard(u)
                S.add(v)
                nk = _coverage(S)
                if nk > best:
                    best = nk
                elif nk == best and random.random() < 0.15:
                    pass
                else:
                    S.discard(v)
                    S.add(u)
    # Fallback with more chords
    for m in (12, 14):
        if time.time() > deadline:
            break
        S = set(random.sample(_CANDIDATES, m))
        best = _coverage(S)
        for _ in range(30000):
            if time.time() > deadline:
                break
            if best == _FULL_MASK:
                sol = _build_solution(S)
                if sol:
                    return sol
            ins = list(S)
            outs = [k for k in _CANDIDATES if k not in S]
            if not ins or not outs:
                break
            u, v = random.choice(ins), random.choice(outs)
            S.discard(u)
            S.add(v)
            nk = _coverage(S)
            if nk >= best:
                best = nk
            else:
                S.discard(v)
                S.add(u)
    sol = _build_solution(S)
    if sol:
        return sol
    # Dense fallback
    S = set(range(2, 39))
    sol = _build_solution(S)
    if sol:
        return sol
    # Absolute last resort
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    cycles = [list(range(L)) for L in range(3, N + 1)]
    return {'edges': edges, 'cycles': cycles}
