"""Rewritten program: find a minimum-coloring of the degree-4 graph such that
the subgraph induced by any two colors is a forest (no bicolored cycle)."""
import time

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _u, _v in _EDGES:
    _ADJ[_u].append(_v)
    _ADJ[_v].append(_u)
for _i in range(_N):
    _ADJ[_i].sort()

def _has_bicolored_cycle(color):
    cu = sorted(set(color))
    m = len(cu)
    for i in range(m):
        a = cu[i]
        for j in range(i + 1, m):
            b = cu[j]
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                    return x
                r = x
                while parent[r] != r:
                    r = parent[r]
                while parent[x] != r:
                    parent[x], x = r, parent[x]
                return r
            for u, v in _EDGES:
                cuu = color[u]; cvv = color[v]
                if (cuu == a and cvv == b) or (cuu == b and cvv == a):
                    ru = find(u); rv = find(v)
                    if ru == rv:
                        return True
                    parent[ru] = rv
    return False

def _valid(color):
    for u, v in _EDGES:
        if color[u] == color[v]:
            return False
    return not _has_bicolored_cycle(color)

def construct():
    start = time.time()
    deadline = start + 82.0
    order = sorted(range(_N), key=lambda i: (-len(_ADJ[i]), i))

    def bt(k, color, colors_used, maxc):
        if time.time() > deadline:
            return None
        if k == _N:
            return color[:]
        v = order[k]
        neigh_colors = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                neigh_colors.add(color[nb])
        # candidate colors 0..maxc-1, not in neighbor colors
        for c in range(0, maxc):
            if c in neigh_colors:
                continue
            color[v] = c
            if c not in colors_used:
                # incremental bicolored-cycle check for new color c vs each existing a
                ok = True
                for a in colors_used:
                    parent = {}
                    def find(x):
                        if x not in parent:
                            parent[x] = x
                            return x
                        r = x
                        while parent[r] != r:
                            r = parent[r]
                        while parent[x] != r:
                            parent[x], x = r, parent[x]
                        return r
                    for u, w in _EDGES:
                        cuu = color[u]; cvv = color[w]
                        if cuu == -1 or cvv == -1:
                            continue
                        if (cuu == a and cvv == c) or (cuu == c and cvv == a):
                            ru = find(u); rv = find(w)
                            if ru == rv:
                                ok = False
                                break
                            parent[ru] = rv
                    if not ok:
                        break
                if not ok:
                    color[v] = -1
                    continue
                colors_used.add(c)
                res = bt(k + 1, color, colors_used, maxc)
                if res is not None:
                    return res
                colors_used.discard(c)
            else:
                res = bt(k + 1, color, colors_used, maxc)
                if res is not None:
                    return res
            color[v] = -1
        return None

    best = None
    for maxc in range(3, 10):
        color = [-1] * _N
        res = bt(0, color, set(), maxc)
        if res is not None:
            best = res
            break
        if time.time() > deadline:
            break

    if best is None:
        best = list(range(_N))

    if not _valid(best):
        best = list(range(_N))

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
