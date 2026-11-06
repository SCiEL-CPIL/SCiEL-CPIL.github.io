"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _is_regular(adj, n, subset):
    mask = 0
    for v in subset:
        mask |= (1 << v)
    d0 = None
    for v in subset:
        d = (adj[v] & mask).bit_count()
        if d0 is None:
            d0 = d
        elif d != d0:
            return False
    return True

def _count_reg_6(adj, n):
    cnt = 0
    for subset in itertools.combinations(range(n), 6):
        if _is_regular(adj, n, subset):
            cnt += 1
    return cnt

def _has_any_regular(adj, n):
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            if _is_regular(adj, n, subset):
                return True
    return False

def _flip(adj, n, u, v):
    if adj[u] & (1 << v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _local_search(adj, n, deadline, max_iter=5000):
    """Local search: flip edges to reduce count of regular 6-subsets."""
    current_cost = _count_reg_6(adj, n)
    if current_cost == 0:
        return True
    
    T = 3.0
    T_min = 0.05
    cooling = 0.998
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        _flip(adj, n, u, v)
        new_cost = _count_reg_6(adj, n)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 1e-10)):
            current_cost = new_cost
        else:
            _flip(adj, n, u, v)
        
        T *= cooling
        
        if current_cost == 0:
            if not _has_any_regular(adj, n):
                return True
            # No 6-subsets regular but larger ones might be; do a few more flips
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                _flip(adj, n, u2, v2)
                if not _has_any_regular(adj, n):
                    return True
            return not _has_any_regular(adj, n)
    
    return False

def _smart_init(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 5
    best_edges = []
    
    # Try n from 20 down to 12
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple densities
        densities = [0.3, 0.4, 0.5, 0.6, 0.7]
        if n <= 15:
            densities = [0.35, 0.45, 0.5, 0.55, 0.65]
        
        found = False
        for p in densities:
            if time.time() > deadline - 5:
                break
            
            adj = _smart_init(n, p)
            
            # Quick check if already valid
            if _count_reg_6(adj, n) == 0 and not _has_any_regular(adj, n):
                edges = _edges_from_adj(adj, n)
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
            
            # Local search
            if _local_search(adj, n, deadline):
                edges = _edges_from_adj(adj, n)
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: empty graph on 5 vertices
    if best_n < 5:
        best_n = 5
        best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
