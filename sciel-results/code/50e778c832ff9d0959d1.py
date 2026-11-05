import time
import random

_START = time.time()
_DEADLINE = _START + 108

def _can_add_edge(u, v, neighbors, n=40):
    # Check if adding (u,v) creates a C6.
    # Path u-a-b-c-d-v
    # a in N(u)
    # b in N(a) \ {u}
    # c in N(b) \ {u, a}
    # d in N(c) \ {u, a, b}
    # v in N(d) \ {u, a, b, c}
    
    # Optimization: Check if v is in N(d) first? No, d depends on c.
    
    # We can prune if the number of remaining steps is greater than the distance to v?
    # But we don't have distance.
    
    for a in neighbors[u]:
        # a is next
        for b in neighbors[a]:
            if b == u:
                continue
            for c in neighbors[b]:
                if c == u or c == a:
                    continue
                for d in neighbors[c]:
                    if d == u or d == a or d == b:
                        continue
                    # Check if v is neighbor of d and v is not visited
                    if v in neighbors[d]:
                        if v != u and v != a and v != b and v != c:
                            return False
    return True

def _greedy_construct(n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    
    neighbors = [[] for _ in range(n)]
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if _can_add_edge(u, v, neighbors, n):
            neighbors[u].append(v)
            neighbors[v].append(u)
            edges.add((u, v))
            
    # Local Search Phase: Try to add non-edges
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if _can_add_edge(u, v, neighbors, n):
                neighbors[u].append(v)
                neighbors[v].append(u)
                edges.add((u, v))
                improved = True
                
    return list(edges)

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = []
    best_count = 0
    
    # Use multiple seeds
    # The fast check allows many more restarts
    for seed in range(10000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges = _greedy_construct(40, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 105:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
