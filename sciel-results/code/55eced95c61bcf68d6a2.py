import time
import random

_START = time.time()
_DEADLINE = 140.0
_N = 18
_MASK_ALL = (1 << _N) - 1

def _elapsed():
    return time.time() - _START

def _is_k4_free_addition(adj, u, v):
    if (adj[u] >> v) & 1:
        return True
    common = adj[u] & adj[v]
    c = common
    while c:
        lsb = c & -c
        x = lsb.bit_length() - 1
        c ^= lsb
        if adj[x] & common:
            return False
    return True

def _build_triangle_adj(adj):
    t_adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common:
                    t_adj[u] |= 1 << v
                    t_adj[v] |= 1 << u
    return t_adj

def _max_independent_set_size(t_adj):
    # Simple B&B for MIS on N=18
    best = [0]
    # Order vertices by degree in t_adj (descending) for better pruning?
    # Or just use the standard recursive approach with bitmasks
    
    def _solve(mask, size):
        if not mask:
            if size > best[0]:
                best[0] = size
            return
        # Prune
        if size + mask.bit_count() <= best[0]:
            return
            
        # Pick a vertex
        # Heuristic: pick vertex with highest degree in remaining graph?
        # For simplicity, pick lowest set bit
        v = (mask & -mask).bit_length() - 1
        
        # Branch 1: Include v
        # Neighbors of v in mask must be removed
        new_mask = mask & ~t_adj[v] & ~(1 << v)
        _solve(new_mask, size + 1)
        
        # Branch 2: Exclude v
        _solve(mask & ~(1 << v), size)

    _solve(_MASK_ALL, 0)
    return best[0]

def _generate_random_k4free():
    adj = [0] * _N
    edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(edges)
    for u, v in edges:
        if _is_k4_free_addition(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj

def construct():
    deadline = _START + _DEADLINE
    best_score = 18
    best_adj = [0] * _N
    
    # Fallback: Turan
    t_adj_turan = _build_triangle_adj([0]*_N) # Placeholder
    # Actually, let's just run the search
    
    random.seed(42)
    
    iterations = 0
    while _elapsed() < deadline - 5:
        adj = _generate_random_k4free()
        t_adj = _build_triangle_adj(adj)
        score = _max_independent_set_size(t_adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            # If we find 10 or lower, we can stop or keep looking for 9
            if best_score <= 10:
                # Try a few more to see if we can get lower, but 10 is likely optimal
                # Let's run a bit more to be sure
                pass
        
        iterations += 1
        if iterations > 5000: break # Safety
        
    # Convert best_adj to edges
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
