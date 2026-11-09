"""Improved: maximum-clique branches + vertex-disjoint paths via greedy DFS."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _degeneracy_order(n, adj):
    deg = [len(adj[v]) for v in range(n)]
    order = []
    for _ in range(n):
        v = min((v for v in range(n) if deg[v] is not None), key=lambda x: deg[x])
        order.append(v)
        deg[v] = None
        for w in adj[v]:
            if deg[w] is not None:
                deg[w] -= 1
    return order

def _tomita_clique(n, adj):
    # Bron-Kerbosch with pivot, on the degeneracy-ordered graph
    best = [0]
    order = _degeneracy_order(n, adj)
    pos = {v: i for i, v in enumerate(order)}

    def expand(R, Pset, Xset):
        if not Pset and not Xset:
            if len(R) > len(best):
                best[:] = R[:]
            return
        # pivot: choose u in Pset ∪ Xset with max |Pset ∩ N(u)|
        cand = Pset | Xset
        if cand:
            u = max(cand, key=lambda x: len(Pset & adj[x]))
            candidates = [v for v in Pset if v not in adj[u]]
        else:
            candidates = list(Pset)
        for v in candidates:
            newR = R + [v]
            expand(newR, Pset & adj[v], Xset & adj[v])
            Pset.remove(v)
            Xset.add(v)

    expand([], set(range(n)), set())
    return best

def _try_build_paths(branches, adj, n, deadline):
    # branches: list of ints (clique). Return list of paths or None.
    if len(branches) < 2:
        return None
    branch_set = set(branches)
    interior = [v for v in range(n) if v not in branch_set]
    # Order pairs: try edges first (direct adjacency)
    pairs = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            a, b = branches[i], branches[j]
            pairs.append((a, b))
    # sort by whether direct edge exists
    pairs.sort(key=lambda ab: 0 if (ab[0] in adj[ab[1]]) else 1)

    used = set()
    paths = {}
    for a, b in pairs:
        if _time.time() > deadline:
            return None
        # find a simple a-b path whose interior avoids used and branches
        # BFS over (node, path) is exponential; use DFS with heuristic
        found = None
        # simple DFS with ordering by distance-to-b
        from collections import deque
        # try shortest path first via BFS
        parent = {a: None}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for nb in adj[cur]:
                if nb in parent:
                    continue
                if nb in branch_set and nb not in (a, b):
                    continue
                if nb in used:
                    continue
                parent[nb] = cur
                q.append(nb)
        if b in parent:
            # reconstruct
            path = [b]
            while path[-1] != a:
                path.append(parent[path[-1]])
            path.reverse()
            found = path
        if found is None:
            return None
        for v in found[1:-1]:
            used.add(v)
        paths[(a, b)] = found
    return [paths[(a, b)] for a, b in pairs]

def construct():
    t0 = _time.time()
    deadline = t0 + 80.0  # safety margin
    adj = _build_adj()
    n = P['n']
    best_clique = _tomita_clique(n, adj)

    # Try the full clique; if path construction fails, drop vertices greedily.
    cur = list(best_clique)
    while len(cur) >= 2 and _time.time() < deadline:
        paths = _try_build_paths(cur, adj, n, deadline)
        if paths is not None:
            return {'branches': list(cur), 'paths': [list(p) for p in paths]}
        # remove the vertex whose removal is most likely to help:
        # try removing each and keep the one that yields a smaller candidate
        # heuristic: remove the last vertex (or the one with fewest non-branch neighbors)
        # We'll just pop the vertex with smallest degree to other branches' neighbors
        # Simple: remove the last element
        cur = cur[:-1]

    # Fallback: just an edge
    a, b = P['edges'][0]
    return {'branches': [int(a), int(b)], 'paths': [[int(a), int(b)]]}
# EVOLVE-BLOCK-END
