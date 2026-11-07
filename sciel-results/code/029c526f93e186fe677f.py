import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = P['edges']

def _build_adj(edge_list):
    adj = [set() for _ in range(_N)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(adj):
    n = _N
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        maxd = 0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 > maxd:
                        maxd = du + 1
                    q.append(v)
        if maxd > diam:
            diam = maxd
    return diam

def _is_connected(adj):
    seen = [False] * _N
    q = [0]
    seen[0] = True
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)

def _valid(added):
    base = set(map(tuple, _BASE))
    es = set()
    for u, v in added:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base:
            return False
        if key in es:
            return False
        es.add(key)
    adj = _build_adj(es)
    if not _is_connected(adj):
        return False
    if _bfs_diameter(adj) > 4:
        return False
    return True

def _try_remove(es, max_iter=500):
    """Try to remove redundant edges while keeping diameter <= 4 and connected."""
    es = set(es)
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        improved = False
        for e in sorted(es):
            es2 = es - {e}
            adj = _build_adj(es2)
            if _is_connected(adj) and _bfs_diameter(adj) <= 4:
                es = es2
                improved = True
                break
    return es

def _greedy_add(es, pool):
    """Add edges from pool (sorted) until diameter <= 4."""
    es = set(es)
    adj = _build_adj(es)
    for e in pool:
        if _bfs_diameter(adj) <= 4:
            break
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
        es.add(e)
    return es

def _candidate_pool():
    """Generate all valid bipartite (even-odd) non-base edges, sorted by descending path distance."""
    base = set(map(tuple, _BASE))
    pool = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in base:
                continue
            if (u ^ v) & 1 == 1:
                pool.append((u, v))
    pool.sort(key=lambda e: -abs(e[0] - e[1]))
    return pool

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    pool = _candidate_pool()
    
    # Greedy: add longest-range edges first until diameter <= 4
    es = _greedy_add(set(), pool)
    
    # Prune: remove any edge that's not needed
    es = _try_remove(es)
    
    # Verify
    if es and _valid(list(es)):
        return [[int(u), int(v)] for (u, v) in sorted(es)]
    
    # Fallback: try a different ordering (shorter edges first)
    pool_short = list(reversed(pool))
    es = _greedy_add(set(), pool_short)
    es = _try_remove(es)
    if es and _valid(list(es)):
        return [[int(u), int(v)] for (u, v) in sorted(es)]
    
    # Last resort: return empty (should not happen)
    return []
# EVOLVE-BLOCK-END
