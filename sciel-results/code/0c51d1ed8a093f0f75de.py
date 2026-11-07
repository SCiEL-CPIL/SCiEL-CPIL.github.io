import time
import random
from collections import deque

def _diameter(n, adj):
    max_dist = 0
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        q = deque([i])
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

def _is_edge_critical(n, adj, edge_set):
    for (u, v) in edge_set:
        new_adj = [set(s) for s in adj]
        new_adj[u].discard(v)
        new_adj[v].discard(u)
        d = _diameter(n, new_adj)
        if d <= 2:
            return False
    return True

def _prune_non_critical(n, adj, edge_set):
    """Remove all non-critical edges."""
    changed = True
    while changed:
        changed = False
        for (u, v) in list(edge_set):
            new_adj = [set(s) for s in adj]
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            d = _diameter(n, new_adj)
            if d <= 2:
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                changed = True
    return edge_set

def construct():
    n = 16
    start = time.time()
    deadline = start + 80
    
    best_edges = set()
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initial graph
        p = 0.3 + 0.05 * restart
        edge_set = set()
        adj = [set() for _ in range(n)]
        for (u, v) in all_edges:
            if random.random() < p:
                edge_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        # Check diameter <= 2
        if _diameter(n, adj) > 2:
            continue
        
        # Prune non-critical edges
        edge_set = _prune_non_critical(n, adj, edge_set)
        
        # Try to add edges back
        candidates = [e for e in all_edges if e not in edge_set]
        random.shuffle(candidates)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for (u, v) in candidates[:]:
                if (u, v) in edge_set:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                if _is_edge_critical(n, adj, edge_set):
                    candidates.remove((u, v))
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edge_set.discard((u, v))
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)
    
    if not best_edges:
        # Fallback to star
        best_edges = set((0, v) for v in range(1, n))
    
    edges = [[u, v] for (u, v) in sorted(best_edges)]
    return edges
