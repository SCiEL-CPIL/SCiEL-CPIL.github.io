import random
import time
from collections import deque

_K = 5
_BASE_V = 10
_BASE_EDGES = [
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
]
_EDGE_IDX = {}
for _i, (_a, _b) in enumerate(_BASE_EDGES):
    _EDGE_IDX[(_a, _b)] = _i
    _EDGE_IDX[(_b, _a)] = _i

# Precompute cycles
def _get_cycles():
    cycles = {4: [], 6: [], 8: []}
    # 4-cycles
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for v1 in range(5, 10):
                for v2 in range(5, 10):
                    if v1 == v2: continue
                    # Walk u1-v1-u2-v2-u1
                    e1 = _EDGE_IDX[(u1, v1)]
                    e2 = _EDGE_IDX[(u2, v1)]
                    e3 = _EDGE_IDX[(u2, v2)]
                    e4 = _EDGE_IDX[(u1, v2)]
                    cycles[4].append((e1, e2, e3, e4))
    # 6-cycles
    # u1-v1-u2-v2-u3-v3-u1
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u3 == u1 or u3 == u2: continue
                for v1 in range(5, 10):
                    for v2 in range(5, 10):
                        if v2 == v1: continue
                        for v3 in range(5, 10):
                            if v3 == v1 or v3 == v2: continue
                            e1 = _EDGE_IDX[(u1, v1)]
                            e2 = _EDGE_IDX[(u2, v1)]
                            e3 = _EDGE_IDX[(u2, v2)]
                            e4 = _EDGE_IDX[(u3, v2)]
                            e5 = _EDGE_IDX[(u3, v3)]
                            e6 = _EDGE_IDX[(u1, v3)]
                            cycles[6].append((e1, e2, e3, e4, e5, e6))
    # 8-cycles
    # u1-v1-u2-v2-u3-v3-u4-v4-u1
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u3 == u1 or u3 == u2: continue
                for u4 in range(5):
                    if u4 == u1 or u4 == u2 or u4 == u3: continue
                    for v1 in range(5, 10):
                        for v2 in range(5, 10):
                            if v2 == v1: continue
                            for v3 in range(5, 10):
                                if v3 == v1 or v3 == v2: continue
                                for v4 in range(5, 10):
                                    if v4 == v1 or v4 == v2 or v4 == v3: continue
                                    e1 = _EDGE_IDX[(u1, v1)]
                                    e2 = _EDGE_IDX[(u2, v1)]
                                    e3 = _EDGE_IDX[(u2, v2)]
                                    e4 = _EDGE_IDX[(u3, v2)]
                                    e5 = _EDGE_IDX[(u3, v3)]
                                    e6 = _EDGE_IDX[(u4, v3)]
                                    e7 = _EDGE_IDX[(u4, v4)]
                                    e8 = _EDGE_IDX[(u1, v4)]
                                    cycles[8].append((e1, e2, e3, e4, e5, e6, e7, e8))
    return cycles

_CYCLES = _get_cycles()

def _check_girth(voltages, m):
    for L in [4, 6, 8]:
        for cyc in _CYCLES[L]:
            s = sum(voltages[e] for e in cyc) % m
            if s == 0:
                return False
    return True

def _find_voltages(m, time_limit=10):
    start_time = time.time()
    n_edges = 25
    while time.time() - start_time < time_limit:
        voltages = [random.randint(0, m-1) for _ in range(n_edges)]
        if _check_girth(voltages, m):
            return voltages
        # Local search
        # ... (omitted for brevity in thought, will add in code)
    return None

def _lift(voltages, m):
    n = 10 * m
    adj = [[] for _ in range(n)]
    for u in range(5):
        for v in range(5, 10):
            e = _EDGE_IDX[(u, v)]
            vol = voltages[e] % m
            for f in range(m):
                a = u * m + f
                b = v * m + (f + vol) % m
                adj[a].append(b)
                adj[b].append(a)
    return n, adj

def _verify_girth(n, adj, g):
    # Non-backtracking BFS
    for s in range(n):
        for v in adj[s]:
            queue = deque()
            queue.append((v, s, s))
            dist = 1
            while queue and dist < g:
                size = len(queue)
                for _ in range(size):
                    node, prev, root = queue.popleft()
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == root:
                            return False
                        queue.append((nb, node, root))
                dist += 1
    return True

def construct(k, g):
    if k == 5 and g == 9:
        # Try voltage graph
        for m in [50, 60, 70, 80, 90, 100, 110, 120]:
            v = _find_voltages(m, time_limit=5)
            if v:
                n, adj = _lift(v, m)
                if _verify_girth(n, adj, 9):
                    edges = []
                    for u in range(n):
                        for w in adj[u]:
                            if u < w:
                                edges.append((u, w))
                    return n, edges
    # Fallback LUW
    q = k
    d = 4
    # ... LUW code ...
    return 1250, [] # Placeholder, will replace with full LUW
