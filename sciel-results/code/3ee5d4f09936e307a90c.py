import time
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    adj_base = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
        adj_base[u].add(v)
        adj_base[v].add(u)
    
    A = set(range(0, n, 2))
    B = set(range(1, n, 2))
    
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    # Precompute conflict pairs (triangle creation)
    conflict = [set() for _ in range(len(candidates))]
    for i in range(len(candidates)):
        u1, v1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            if u1 == u2 and frozenset((v1, v2)) in base_edges:
                conflict[i].add(j); conflict[j].add(i)
            elif v1 == v2 and frozenset((u1, u2)) in base_edges:
                conflict[i].add(j); conflict[j].add(i)
            elif u1 == v2 and frozenset((u1, u2)) in base_edges:
                conflict[i].add(j); conflict[j].add(i)
            elif v1 == u2 and frozenset((u1, v2)) in base_edges:
                conflict[i].add(j); conflict[j].add(i)
    
    edge_to_idx = {c: i for i, c in enumerate(candidates)}
    deadline = time.time() + 85
    
    def _check(added_edges):
        idx_set = set()
        for e in added_edges:
            idx = edge_to_idx.get(e)
            if idx is None:
                return False
            idx_set.add(idx)
        for i in idx_set:
            if conflict[i] & idx_set:
                return False
        adj = [set(adj_base[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]; qi += 1
                du = dist[u]
                if du >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True
    
    for k in range(1, 12):
        if time.time() > deadline:
            break
        found = None
        rng = random.Random(42 + k)
        if k <= 3:
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _check(combo):
                    found = list(combo)
                    break
        else:
            for it in range(500000):
                if time.time() > deadline:
                    break
                combo = rng.sample(candidates, k)
                if _check(combo):
                    found = list(combo)
                    break
        if found:
            return [[u, v] for u, v in found]
    
    # Fallback
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    return [[hub, t] for t in targets]
