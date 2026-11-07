import time
import random
from collections import Counter

N = 15
DEADLINE = 75.0

def _is_valid(adj, n):
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        seen = set()
        for v in adj[u]:
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def construct():
    start = time.time()
    best_adj = [set() for _ in range(N)]
    best_count = 0
    
    # Initialize with Star
    for v in range(1, N):
        best_adj[0].add(v)
        best_adj[v].add(0)
    best_count = N - 1
    
    # Try adding one leaf-leaf edge
    for i in range(1, N):
        for j in range(i+1, N):
            if j not in best_adj[i]:
                best_adj[i].add(j)
                best_adj[j].add(i)
                if _is_valid(best_adj, N):
                    cnt = _edge_count(best_adj, N)
                    if cnt > best_count:
                        best_count = cnt
                        # Keep this as current best
                    else:
                        best_adj[i].discard(j)
                        best_adj[j].discard(i)
                else:
                    best_adj[i].discard(j)
                    best_adj[j].discard(i)

    # Local Search
    current_adj = [set(s) for s in best_adj]
    current_count = best_count
    improved = True
    while improved and time.time() - start < DEADLINE:
        improved = False
        # Try adding edges
        for i in range(N):
            for j in range(i+1, N):
                if j not in current_adj[i]:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                    if _is_valid(current_adj, N):
                        cnt = _edge_count(current_adj, N)
                        if cnt > best_count:
                            best_count = cnt
                            best_adj = [set(s) for s in current_adj]
                            improved = True
                        else:
                            current_adj[i].discard(j)
                            current_adj[j].discard(i)
                    else:
                        current_adj[i].discard(j)
                        current_adj[j].discard(i)
        
        # Try swapping edges
        # Simple swap: remove (u,v), add (x,y)
        # This is expensive, so we do random restarts of add-only if stuck?
        # Or just random perturbation.
        
    # Convert to list
    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
