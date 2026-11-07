import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85

    # Build adjacency from base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)

    existing = set()
    for u, v in base_edges:
        existing.add((min(u, v), max(u, v)))

    # All possible non-path edges (bipartite only to guarantee triangle-free)
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing and u % 2 != v % 2:
                all_edges.append((u, v))

    # Verify a set of added edges
    def verify(added):
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            if adj2[u] & adj2[v]:
                return False
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Check connected
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj2[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        # Check diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
        return True

    # Precompute: for each vertex, the set of vertices at path-distance 2
    dist2 = [set() for _ in range(n)]
    for s in range(n):
        for d in [-2, 2]:
            t = s + d
            if 0 <= t < n:
                dist2[s].add(t)

    best = None
    best_count = float('inf')

    # Strategy 1: Try hub-based constructions (single hub)
    for h in range(n):
        if time.time() > deadline:
            break
        added = []
        for t in sorted(dist2[h]):
            edge = (min(h, t), max(h, t))
            if edge not in existing:
                added.append(list(edge))
        
        if len(added) < best_count and verify(added):
            best = added
            best_count = len(added)

    # Strategy 2: Two-hub constructions
    for h1 in range(n):
        for h2 in range(h1 + 2, n):
            if time.time() > deadline:
                break
            added = []
            for t in sorted(dist2[h1]):
                edge = (min(h1, t), max(h1, t))
                if edge not in existing:
                    added.append(list(edge))
            for t in sorted(dist2[h2]):
                edge = (min(h2, t), max(h2, t))
                if edge not in existing:
                    if edge not in [tuple(e) for e in added]:
                        added.append(list(edge))
            
            if len(added) < best_count and verify(added):
                best = added
                best_count = len(added)

    # Strategy 3: Greedy with local search reduction
    def greedy_build():
        added = set()
        adj_local = [set(adj[i]) for i in range(n)]
        
        while True:
            if time.time() > deadline:
                break
            
            # Compute all-pairs distances
            all_dists = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj_local[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                all_dists.append(dist)
            
            # Count far pairs
            far_count = sum(1 for i in range(n) for j in range(i+1, n) if all_dists[i][j] > 4)
            if far_count == 0:
                break
            
            # Find best edge to add
            best_edge = None
            best_far_count = far_count
            
            for u, v in all_edges:
                if (u, v) in added:
                    continue
                # Temporarily add and count far pairs
                new_adj = [set(adj_local[i]) for i in range(n)]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                new_far = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    q = deque([src])
                    while q:
                        node = q.popleft()
                        for nb in new_adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                q.append(nb)
                    for j in range(src+1, n):
                        if dist[j] > 4:
                            new_far += 1
                
                if new_far < best_far_count:
                    best_far_count = new_far
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            added.add(best_edge)
            adj_local = [set(adj[i]) for i in range(n)]
            for u, v in base_edges:
                adj_local[u].add(v)
                adj_local[v].add(u)
            for u, v in added:
                adj_local[u].add(v)
                adj_local[v].add(u)
        
        return added

    def local_search_reduce(added_edges):
        """Try to remove edges while maintaining validity."""
        added = set(added_edges)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added):
                if time.time() > deadline:
                    break
                new_set = added - {edge}
                if verify(list(new_set)):
                    added = new_set
                    improved = True
                    break
        return added

    # Run greedy build
    greedy_result = greedy_build()
    if greedy_result:
        greedy_result = local_search_reduce(greedy_result)
        if verify(list(greedy_result)) and len(greedy_result) < best_count:
            best = list(greedy_result)
            best_count = len(best)

    # Strategy 4: Try independent sets of size 3 and 4
    def gen_independent_sets(n, k, start=0):
        if k == 0:
            yield []
            return
        if start >= n:
            return
        # Include start
        for rest in gen_independent_sets(n, k - 1, start + 2):
            yield [start] + rest
        # Exclude start
        for rest in gen_independent_sets(n, k, start + 1):
            yield rest

    for k in [3, 4]:
        if time.time() > deadline:
            break
        for S in gen_independent_sets(n, k):
            if time.time() > deadline:
                break
            added = []
            for s in S:
                for t in sorted(dist2[s]):
                    edge = (min(s, t), max(s, t))
                    if edge not in existing:
                        if edge not in [tuple(e) for e in added]:
                            added.append(list(edge))
            
            if len(added) < best_count and verify(added):
                best = added
                best_count = len(added)

    # If we haven't found anything, use a known valid solution
    if best is None:
        # Fallback: hub at 11 with specific connections
        best = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
        best_count = len(best)

    return [[int(u), int(v)] for u, v in best]
