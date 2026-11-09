"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

import time
import random

def construct():
    n = P['n']
    edges = P['edges']
    start = time.time()
    DEADLINE = start + 80.0

    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    pair_uf = {}
    color = [-1] * n

    def _pair_key(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return (c1, c2)

    def _find(d, x):
        while d[x] != x:
            d[x] = d[d[x]]
            x = d[x]
        return x

    def _union(d, x, y):
        rx, ry = _find(d, x), _find(d, y)
        if rx == ry:
            return False
        d[rx] = ry
        return True

    def _add_vertex(v, c):
        color[v] = c
        for nb in adj[v]:
            cn = color[nb]
            if cn == -1:
                continue
            if cn == c:
                return False
            k = _pair_key(c, cn)
            d = pair_uf.get(k)
            if d is None:
                d = {}
                pair_uf[k] = d
            if v not in d:
                d[v] = v
            if nb not in d:
                d[nb] = nb
            if _find(d, v) == _find(d, nb):
                return False
            _union(d, v, nb)
        return True

    def _remove_vertex(v):
        c = color[v]
        color[v] = -1
        affected = set()
        for nb in adj[v]:
            cn = color[nb]
            if cn != -1 and cn != c:
                affected.add(_pair_key(c, cn))
        for k in affected:
            d = pair_uf.get(k)
            if d is None:
                continue
            c1, c2 = k
            nd = {}
            for u in range(n):
                if color[u] == c1 or color[u] == c2:
                    nd[u] = u
            for u in range(n):
                if color[u] == c1 or color[u] == c2:
                    for nb in adj[u]:
                        if color[nb] in (c1, c2) and color[nb] != color[u]:
                            if u in nd and nb in nd:
                                _union(nd, u, nb)
            pair_uf[k] = nd

    def _available(v, maxc):
        avail = []
        for c in range(maxc):
            bad = False
            for nb in adj[v]:
                if color[nb] == c:
                    bad = True
                    break
            if bad:
                continue
            ok = True
            for nb in adj[v]:
                cn = color[nb]
                if cn == -1 or cn == c:
                    continue
                k = _pair_key(c, cn)
                d = pair_uf.get(k)
                if d is not None:
                    if v in d and nb in d and _find(d, v) == _find(d, nb):
                        ok = False
                        break
            if ok:
                avail.append(c)
        return avail

    def _mrv_unassigned(maxc):
        best = -1
        best_cnt = 10**9
        for v in range(n):
            if color[v] != -1:
                continue
            cnt = len(_available(v, maxc))
            if cnt < best_cnt:
                best_cnt = cnt
                best = v
                if cnt == 0:
                    break
        return best, best_cnt

    def _bt(maxc):
        if time.time() > DEADLINE:
            return None
        v, cnt = _mrv_unassigned(maxc)
        if v == -1:
            return list(color)
        if cnt == 0:
            return None
        avail = _available(v, maxc)
        used_colors = set(color)
        avail.sort(key=lambda c: (0 if c in used_colors else 1, c))
        for c in avail:
            if _add_vertex(v, c):
                res = _bt(maxc)
                if res is not None:
                    return res
            _remove_vertex(v)
        return None

    best_solution = list(range(n))  # trivial valid
    best_maxc = n

    for maxc in range(4, n):
        if time.time() > DEADLINE:
            break
        # reset state
        for i in range(n):
            color[i] = -1
        pair_uf.clear()
        found = _bt(maxc)
        if found is not None:
            best_solution = found
            best_maxc = maxc
            break

    sol = best_solution
    used = sorted(set(sol))
    remap = {c: i for i, c in enumerate(used)}
    out = [int(remap[c]) for c in sol]
    return out

# EVOLVE-BLOCK-END
