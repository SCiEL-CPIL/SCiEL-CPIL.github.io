"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 16}

def _bfs_diameter(adj, n):
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if -1 in dist:
            return float('inf')
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_valid(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    if _bfs_diameter(adj, n) != 2:
        return False
    for u, v in edges:
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        if _bfs_diameter(adj2, n) == 2:
            return False
    return True

def construct():
    n = P['n']
    
    # The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4).
    # For n=16, this is 64. The unique extremal graph is K_{8,8}.
    # K_{8,8} has diameter 2.
    # Every edge in K_{8,8} is critical: removing (u,v) increases d(u,v) to 3, increasing diameter to 3.
    # Any graph with more edges must contain a triangle, and edges in triangles are generally not critical
    # (removing one leaves a path of length 2 via the third vertex, so diameter might not increase).
    # Thus, K_{n/2, n/2} is the optimal solution for even n >= 4.
    
    a = n // 2
    b = n - a
    
    edges = []
    for i in range(a):
        for j in range(a, n):
            edges.append([i, j])
            
    # Verify validity just in case, though it is mathematically proven for this structure
    if _is_valid(edges, n):
        return [[int(u), int(v)] for u, v in edges]
    
    # Fallback to previous logic if something unexpected happens (should not happen for n=16)
    deadline = time.time() + 85
    random.seed(42)
    best_edges = [[0, i] for i in range(1, n)]
    best_count = len(best_edges)
    current = list(best_edges)
    edge_set = set((min(u,v), max(u,v)) for u,v in current)
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        candidates = [(i,j) for i,j in all_pairs if (i,j) not in edge_set]
        random.shuffle(candidates)
        added_any = False
        for u, v in candidates:
            if time.time() > deadline:
                break
            new_edges = current + [[u, v]]
            if _is_valid(new_edges, n):
                current = new_edges
                edge_set.add((u, v))
                added_any = True
                if len(current) > best_count:
                    best_edges = list(current)
                    best_count = len(best_edges)
        if not added_any:
            if len(current) >= 3:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(current)-1)
                    removed = current.pop(idx)
                    edge_set.discard((min(removed[0],removed[1]), max(removed[0],removed[1])))
                    candidates2 = [(i,j) for i,j in all_pairs if (i,j) not in edge_set]
                    random.shuffle(candidates2)
                    swapped = False
                    for u, v in candidates2[:50]:
                        new_edges = current + [[u, v]]
                        if _is_valid(new_edges, n):
                            current = new_edges
                            edge_set.add((u, v))
                            swapped = True
                            if len(current) > best_count:
                                best_edges = list(current)
                                best_count = len(best_edges)
                            break
                    if not swapped:
                        current.append(removed)
                        edge_set.add((min(removed[0],removed[1]), max(removed[0],removed[1])))
    return [[int(u), int(v)] for u, v in best_edges]
# EVOLVE-BLOCK-END
