import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin (limit 90s)
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Precompute all possible non-base edges
    all_nonbase = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) not in base_edges:
                all_nonbase.append((u,v))
    
    # For fast triangle check, precompute for each pair (u,v) the set of common path-neighbors
    # A new edge (u,v) is triangle-free w.r.t. current adj iff no current neighbor w of u is also neighbor of v
    # We'll check dynamically.
    
    def make_adj(added):
        adj = [set() for _ in range(n)]
        for u,v in base_edges:
            adj[u].add(v); adj[v].add(u)
        for u,v in added:
            adj[u].add(v); adj[v].add(u)
        return adj
    
    def is_triangle_free(adj, u, v):
        # check if adding (u,v) creates a triangle
        # common neighbor of u and v?
        if len(adj[u]) < len(adj[v]):
            for w in adj[u]:
                if w != v and w in adj[v]:
                    return False
        else:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
        return True
    
    def diameter_le_4(adj):
        # BFS from each vertex to depth 4, check all reached
        for s in range(n):
            dist = [-1]*n
            dist[s] = 0
            q = deque([s])
            while q:
                node = q.popleft()
                if dist[node] == 4:
                    continue
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node]+1
                        q.append(nb)
            for d in dist:
                if d == -1:
                    return False
        return True
    
    def full_diameter(adj):
        maxd = 0
        for s in range(n):
            dist = [-1]*n
            dist[s]=0
            q=deque([s])
            while q:
                node=q.popleft()
                for nb in adj[node]:
                    if dist[nb]==-1:
                        dist[nb]=dist[node]+1
                        q.append(nb)
            for d in dist:
                if d>maxd: maxd=d
        return maxd
    
    # Greedy construction
    def greedy():
        adj = make_adj([])
        added = []
        added_set = set()
        while True:
            if time.time() > deadline:
                break
            if diameter_le_4(adj):
                break
            best = None
            best_diam = full_diameter(adj) + 1
            for u,v in all_nonbase:
                if (u,v) in added_set: continue
                if not is_triangle_free(adj, u, v): continue
                adj[u].add(v); adj[v].add(u)
                d = full_diameter(adj)
                adj[u].discard(v); adj[v].discard(u)
                if d < best_diam:
                    best_diam = d
                    best = (u,v)
            if best is None:
                break
            u,v = best
            adj[u].add(v); adj[v].add(u)
            added_set.add((u,v))
            added.append((u,v))
        return added
    
    # Local search to minimize
    def local_search(added, time_limit):
        added_set = set(added)
        adj = make_adj(added)
        best_added = list(added)
        best_len = len(added)
        # First: try removing each edge
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for e in list(added_set):
                u,v = e
                adj[u].discard(v); adj[v].discard(u)
                if diameter_le_4(adj):
                    added_set.discard(e)
                    best_added = list(added_set)
                    best_len = len(added_set)
                    changed = True
                else:
                    adj[u].add(v); adj[v].add(u)
        # Simulated annealing / iterated: try swaps
        # ...
        return best_added
    
    # Run
    random.seed(42)
    added = greedy()
    # verify
    adj = make_adj(added)
    if not (diameter_le_4(adj)):
        # fallback: return something feasible? 
        pass
    added = local_search(added, deadline)
    adj = make_adj(added)
    # ensure feasible
    if not diameter_le_4(adj):
        # try greedy again or return greedy result
        added = greedy()
    
    return [[int(u), int(v)] for u,v in added]
