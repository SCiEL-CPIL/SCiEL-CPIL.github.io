"""K3,3-free dense graph on 20 vertices - maximize edges."""
import time
import itertools
import random

def construct():
    deadline = time.time() + 140
    n = 20
    
    def can_add(adj, u, v):
        for a in range(n):
            if a == u or a == v:
                continue
            if v not in adj[a]:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if v not in adj[b]:
                    continue
                if len(adj[u] & adj[a] & adj[b]) >= 2:
                    return False
        for a in range(n):
            if a == u or a == v:
                continue
            if u not in adj[a]:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if u not in adj[b]:
                    continue
                if len(adj[v] & adj[a] & adj[b]) >= 2:
                    return False
        return True
    
    def greedy_build(adj, edge_order):
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if v not in adj[u] and can_add(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
    
    def edge_count(adj):
        return sum(len(s) for s in adj) // 2
    
    best_adj = None
    best_count = 0
    
    for _ in range(100):
        if time.time() > deadline:
            break
        adj = [set() for _ in range(n)]
        edge_order = list(itertools.combinations(range(n), 2))
        random.shuffle(edge_order)
        greedy_build(adj, edge_order)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]
    
    if best_adj is None:
        best_adj = [set() for _ in range(n)]
    
    adj = best_adj
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u, v in itertools.combinations(range(n), 2):
            if time.time() > deadline:
                break
            if v not in adj[u] and can_add(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                improved = True
        if improved:
            continue
        
        found_swap = False
        existing = [(u, v) for u in range(n) for v in adj[u] if u < v]
        random.shuffle(existing)
        for u, v in existing:
            if time.time() > deadline:
                break
            adj[u].discard(v)
            adj[v].discard(u)
            added = 0
            for u2, v2 in itertools.combinations(range(n), 2):
                if time.time() > deadline:
                    break
                if v2 not in adj[u2] and can_add(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    added += 1
                    if added >= 2:
                        break
            if added >= 2:
                found_swap = True
                break
            else:
                adj[u].add(v)
                adj[v].add(u)
        if found_swap:
            improved = True
    
    edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
    return edges
