"""Rewritten: search for a large complete-graph subdivision (max branch vertices)."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
ADJ = [set() for _ in range(N)]
for a, b in P['edges']:
    ADJ[a].add(b)
    ADJ[b].add(a)

DEADLINE = time.time() + 80.0

def _bipartite_max_matching(L, R, adjL):
    """Kuhn: max matching covering L (|L|<=|R|). adjL: dict u->list of R nodes."""
    matchR = {v: -1 for v in R}
    def try_kuhn(u, seen):
        for v in adjL.get(u, ()):
            if seen.get(v):
                continue
            seen[v] = 1
            if matchR[v] == -1 or try_kuhn(matchR[v], seen):
                matchR[v] = u
                return True
        return False
    for u in L:
        if not try_kuhn(u, {}):
            return False
    return True

def _subdivision_paths(branches):
    """Return list of (u,v,path) for each pair, or None if no internally-disjoint routing."""
    B = set(branches)
    pairs = list(combinations(branches, 2))
    # candidate simple paths per pair (BFS, bounded length)
    cand = {}
    for u, v in pairs:
        found = []
        seen = {u: 0}
        q = deque([(u, [u])])
        while q and len(found) < 12:
            x, path = q.popleft()
            if len(path) >= 12:
                continue
            if x == v:
                found.append(list(path))
                continue
            for y in ADJ[x]:
                if y in B and y != v:
                    continue
                if y in seen:
                    continue
                seen[y] = len(path) + 1
                q.append((y, path + [y]))
        if not found:
            return None
        cand[(u, v)] = found
    # nodes = internal edges of candidate paths
    node_set = []
    node_id = {}
    for (u, v), lst in cand.items():
        for path in lst:
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i])
                if e not in node_id:
                    node_id[e] = len(node_set)
                    node_set.append(e)
    # left = pairs, right = nodes
    L = list(cand.keys())
    R = list(node_id.values())
    adjL = {}
    for (u, v) in L:
        s = set()
        for path in cand[(u, v)]:
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i])
                s.add(node_id[e])
        adjL[(u, v)] = list(s)
    if not _bipartite_max_matching(L, R, adjL):
        return None
    # extract paths
    matchR = {v: -1 for v in R}
    def try_kuhn(u, seen):
        for v in adjL.get(u, ()):
            if seen.get(v):
                continue
            seen[v] = 1
            if matchR[v] == -1 or try_kuhn(matchR[v], seen):
                matchR[v] = u
                return True
        return False
    for u in L:
        try_kuhn(u, {})
    # invert: node -> pair
    node_pair = {}
    for v, u in matchR.items():
        if u != -1:
            node_pair[v] = u
    paths = {}
    for (u, v) in L:
        used_edges = [e for e in node_set if node_pair.get(node_id[e]) == (u, v)]
        # reconstruct path: connect u to v using used edges (a simple path)
        # build adjacency among used edges
        adj2 = {u: []}
        for a, b in used_edges:
            adj2.setdefault(a, []).append(b)
            adj2.setdefault(b, []).append(a)
        # walk from u
        path = [u]
        cur = u
        while cur != v:
            nxts = [z for z in adj2.get(cur, []) if z not in path or z == v]
            if not nxts:
                return None
            cur = nxts[0]
            path.append(cur)
        if path[0] != u or path[-1] != v:
            return None
        paths[(u, v)] = path
    out = []
    for (u, v) in pairs:
        out.append((u, v, paths[(u, v)]))
    return out

def construct():
    best = None
    def consider(b):
        nonlocal best
        r = _subdivision_paths(b)
        if r is None:
            return False
        if best is None or len(b) > len(best[0]):
            best = (b, r)
        return True
    # initial: a triangle
    tri = None
    for a in range(N):
        for b in range(a + 1, N):
            if b not in ADJ[a]:
                continue
            for c in range(b + 1, N):
                if c in ADJ[a] and c in ADJ[b]:
                    tri = [a, b, c]
                    break
            if tri:
                break
        if tri:
            break
    if tri is None:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    consider(tri)
    branches = list(tri)
    # greedy grow
    while time.time() < DEADLINE:
        B = set(branches)
        added = False
        for w in range(N):
            if w in B or time.time() > DEADLINE:
                continue
            cand = branches + [w]
            if consider(cand):
                branches = cand
                added = True
                break
        if not added:
            break
    if best is None:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    b, r = best
    paths = []
    for (u, v, p) in r:
        paths.append([int(x) for x in p])
    return {'branches': [int(x) for x in b], 'paths': paths}
# EVOLVE-BLOCK-END
