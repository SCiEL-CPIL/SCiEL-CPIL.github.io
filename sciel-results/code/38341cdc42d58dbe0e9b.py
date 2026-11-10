import random

def _build_bipartite_perms(nhalf, k, rng):
    # nhalf=6, k=3
    M = [[0]*nhalf for _ in range(nhalf)]
    for _ in range(k):
        perm = list(range(nhalf))
        rng.shuffle(perm)
        for i in range(nhalf):
            M[i][perm[i]] = 1
    edges = []
    for i in range(nhalf):
        for j in range(nhalf):
            if M[i][j]:
                edges.append((i, nhalf + j))
    return edges

def _adj_from_edges(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _cycle_lengths(adj, n, cap=500000):
    lengths = set()
    count = 0
    for start in range(n):
        stack = [(start, -1, [start], {start})]
        while stack:
            u, prev, path, visited = stack.pop()
            for v in sorted(adj[u]):
                if v == prev:
                    continue
                if v == start:
                    if len(path) >= 4:
                        lengths.add(len(path))
                        count += 1
                        if count > cap:
                            return lengths
                elif v not in visited and v > start:
                    stack.append((v, u, path + [v], visited | {v}))
    return lengths

def _valid(edges, n):
    # simple, no loops
    seen = set()
    for u, v in edges:
        if u == v: return False
        if not (0 <= u < n and 0 <= v < n): return False
        key = (min(u,v), max(u,v))
        if key in seen: return False
        seen.add(key)
    adj = _adj_from_edges(edges, n)
    for i in range(n):
        if len(adj[i]) < 3: return False
    # triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # has a cycle
    if not _cycle_lengths(adj, n):
        return False
    return True

def construct():
    n = 12
    best = None
    best_len = None
    rng = random.Random(12345)
    candidates = []
    # fixed candidates: a few specific permutation sets
    fixed = [
        [0,1,2,3,4,5],  # identity
    ]
    # generate many random
    for trial in range(400):
        edges = _build_bipartite_perms(6, 3, rng)
        candidates.append(edges)
    for edges in candidates:
        if not _valid(edges, n):
            continue
        adj = _adj_from_edges(edges, n)
        L = _cycle_lengths(adj, n)
        if not L:
            continue
        k = len(L)
        if best_len is None or k < best_len:
            best_len = k
            best = edges
        if k == 1:
            break
    if best is None:
        # fallback: seed-like graph
        best = [(i,(i+1)%6) for i in range(6)] + [(i+6,(i+1)%6+6) for i in range(6)] + [(i,i+6) for i in range(6)]
    out = []
    seen = set()
    for u, v in best:
        key = (min(u,v), max(u,v))
        if key not in seen:
            seen.add(key)
            out.append([int(u), int(v)])
    return out
