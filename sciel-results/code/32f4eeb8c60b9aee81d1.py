"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}
_DEADLINE = 85

def _is_valid_fast(edges, n):
    """Fast validity check for n=15."""
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 3-edge paths s-u-v-t
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _greedy_build(n, seed=0):
    rng = random.Random(seed)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    edges = []
    for u, v in all_edges:
        edges.append([u, v])
        if not _is_valid_fast(edges, n):
            edges.pop()
    return edges

def _local_search(edges, n, seed=0, max_iter=2000):
    rng = random.Random(seed)
    best = [e[:] for e in edges]
    best_count = len(best)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        current_set = set()
        for u, v in best:
            current_set.add((min(u,v), max(u,v)))
        missing = [p for p in all_pairs if p not in current_set]
        
        if missing:
            u, v = rng.choice(missing)
            new_edges = best + [[u, v]]
            if _is_valid_fast(new_edges, n):
                best = new_edges
                best_count += 1
    
    return best

def construct():
    n = P['n']
    start = time.time()
    
    best_edges = None
    best_count = 0
    
    for seed in range(30):
        if time.time() > _DEADLINE:
            break
        edges = _greedy_build(n, seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    if best_edges is not None:
        best_edges = _local_search(best_edges, n, seed=123)
        best_count = len(best_edges)
    
    return [[int(u), int(v)] for u, v in best_edges]
# EVOLVE-BLOCK-END
