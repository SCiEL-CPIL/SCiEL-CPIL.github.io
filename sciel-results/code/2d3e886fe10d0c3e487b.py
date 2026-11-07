import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = P['edges']
_BASE_SET = set((min(u,v), max(u,v)) for u,v in _BASE)

def _bipartite_ok(u, v):
    return (u ^ v) & 1 == 1

def _build_adj(edge_set):
    adj = [set() for _ in range(_N)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _diameter_leq_4(adj):
    n = _N
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[u]
            if du >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
        for i in range(n):
            if dist[i] == -1 or dist[i] > 4:
                return False
    return True

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
    es = set()
    for u, v in added:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in _BASE_SET:
            return False
        if key in es:
            return False
        es.add(key)
    adj = _build_adj(es)
    if not _is_connected(adj):
        return False
    if not _diameter_leq_4(adj):
        return False
    return True

def _candidate_pool():
    pool = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in _BASE_SET:
                continue
            if _bipartite_ok(u, v):
                pool.append((u, v))
    return pool

def _greedy_build():
    n = _N
    es = set()
    adj = [set() for _ in range(n)]
    pool = _candidate_pool()
    pool.sort(key=lambda e: -abs(e[0] - e[1]))
    for e in pool:
        if _diameter_leq_4(adj):
            break
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
        es.add(e)
    improved = True
    while improved:
        improved = False
        for e in list(es):
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            if _is_connected(adj) and _diameter_leq_4(adj):
                es.discard(e)
                improved = True
                break
            else:
                adj[u].add(v)
                adj[v].add(u)
    return es

def _random_restart_search(deadline, max_trials=80):
    best = None
    best_size = 999
    rng = random.Random(42)
    pool = _candidate_pool()
    long_edges = [e for e in pool if abs(e[0] - e[1]) >= 6]
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        es = set()
        k = rng.randint(4, 8)
        for e in rng.sample(long_edges, min(k, len(long_edges))):
            es.add(e)
        
        adj = _build_adj(es)
        if not _diameter_leq_4(adj):
            for e in pool:
                if e in es:
                    continue
                if _diameter_leq_4(adj):
                    break
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
                es.add(e)
        
        improved = True
        while improved:
            improved = False
            for e in list(es):
                u, v = e
                adj2 = _build_adj(es - {e})
                if _is_connected(adj2) and _diameter_leq_4(adj2):
                    es.discard(e)
                    improved = True
                    break
        
        if len(es) < best_size:
            best = es
            best_size = len(es)
            if best_size <= 5:
                break
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    
    es = _greedy_build()
    best = es
    best_size = len(es)
    
    small = _random_restart_search(deadline - 5)
    if small is not None and len(small) < best_size:
        best = small
        best_size = len(small)
    
    rng = random.Random(77)
    pool = _candidate_pool()
    pool.sort(key=lambda e: -abs(e[0] - e[1]))
    for trial in range(30):
        if time.time() > deadline - 3:
            break
        es_copy = set(best)
        if es_copy:
            e_remove = rng.choice(list(es_copy))
            es_copy.discard(e_remove)
            for e_add in pool[:30]:
                if e_add in es_copy:
                    continue
                es_try = es_copy | {e_add}
                adj = _build_adj(es_try)
                if _is_connected(adj) and _diameter_leq_4(adj):
                    es_copy = es_try
                    break
        improved = True
        while improved:
            improved = False
            for e in list(es_copy):
                u, v = e
                adj = _build_adj(es_copy - {e})
                if _is_connected(adj) and _diameter_leq_4(adj):
                    es_copy.discard(e)
                    improved = True
                    break
        if len(es_copy) < best_size:
            best = es_copy
            best_size = len(es_copy)
    
    if best and _valid(list(best)):
        return [[int(u), int(v)] for (u, v) in sorted(best)]
    
    es = _greedy_build()
    if es and _valid(list(es)):
        return [[int(u), int(v)] for (u, v) in sorted(es)]
    
    return []
# EVOLVE-BLOCK-END
