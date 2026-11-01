"""Voltage-graph lift of the Petersen graph over Z_m (Thue-Morse driven) for the cage problem."""
import time
import random

# EVOLVE-BLOCK-START

_PET_ADJ = [
    [1, 4, 5],
    [0, 3, 6],
    [0, 2, 7],
    [1, 3, 8],
    [0, 4, 9],
    [0, 5, 7],
    [1, 6, 8],
    [2, 5, 9],
    [3, 6, 9],
    [4, 7, 8],
]
_PET_EDGES = []
for _u in range(10):
    for _v in _PET_ADJ[_u]:
        if _u < _v:
            _PET_EDGES.append((_u, _v))
assert len(_PET_EDGES) == 15
_EDGE_IDX = {e: i for i, e in enumerate(_PET_EDGES)}


def _thue_morse(n):
    a = bytearray(n)
    for i in range(1, n):
        a[i] = a[i >> 1] ^ (i & 1)
    return a


def _enumerate_closed_nb_walks(g):
    walks = []
    for start in range(10):
        stack = [(start, -1, [start], 0)]
        while stack:
            cur, prev, path, length = stack.pop()
            if length >= g - 1:
                continue
            for nxt in _PET_ADJ[cur]:
                if nxt == prev:
                    continue
                newlen = length + 1
                newpath = path + [nxt]
                if newlen >= 3 and nxt == start:
                    walks.append(newpath)
                if newlen < g - 1:
                    stack.append((nxt, cur, newpath, newlen))
    return walks


_CLOSED_WALKS_CACHE = {}


def _get_closed_walks(g):
    if g not in _CLOSED_WALKS_CACHE:
        _CLOSED_WALKS_CACHE[g] = _enumerate_closed_nb_walks(g)
    return _CLOSED_WALKS_CACHE[g]


def _voltage_sum_zero(m, volt, walks):
    for path in walks:
        s = 0
        L = len(path) - 1
        for i in range(L):
            u = path[i]
            v = path[i + 1]
            e = _EDGE_IDX.get((u, v))
            if e is None:
                e = _EDGE_IDX[(v, u)]
                s -= volt[e]
            else:
                s += volt[e]
        if s % m == 0:
            return True
    return False


def _build_lift(m, volt):
    n = 10 * m
    adj = [[] for _ in range(n)]
    for (u, v), w in zip(_PET_EDGES, volt):
        w = w % m
        for a in range(m):
            iu = u * m + a
            iv = v * m + (a + w) % m
            adj[iu].append(iv)
            adj[iv].append(iu)
    return adj


def construct(k, g):
    deadline = time.time() + 110
    walks = _get_closed_walks(g)
    base_tm = _thue_morse(300)
    rng = random.Random(777)
    best = None
    best_n = float('inf')
    m = 40
    while m < 1000:
        if time.time() > deadline:
            break
        found_for_m = False
        for trial in range(200):
            if time.time() > deadline:
                break
            volt = []
            for e in range(15):
                base = base_tm[e * 4 + trial % 11] * (m // 2)
                volt.append((base + rng.randint(0, m - 1)) % m)
            if _voltage_sum_zero(m, volt, walks):
                continue
            n = 10 * m
            if n < best_n:
                best_n = n
                adj = _build_lift(m, volt)
                best = (n, adj)
            found_for_m = True
            break
        if found_for_m:
            break
        m += 1
    if best is None:
        # Fallback: random 3-regular graph on large n
        n = 2 ** (g - 1) + 500
        n = (n // 2) * 2
        rng2 = random.Random(4242)
        for attempt in range(30):
            if time.time() > deadline:
                break
            half = 3 * n
            stubs = []
            for v in range(n):
                stubs.extend([v, v, v])
            perm = list(range(half))
            rng2.shuffle(perm)
            adj = [[] for _ in range(n)]
            ok = True
            for i in range(0, half, 2):
                u = stubs[perm[i]]
                v = stubs[perm[i + 1]]
                if u == v or v in adj[u]:
                    ok = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not ok:
                continue
            best = (n, adj)
            break
    n, adj = best
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return n, edges

# EVOLVE-BLOCK-END
