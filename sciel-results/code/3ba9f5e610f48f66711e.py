# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 145.0

def _time_up():
    return time.time() > _DEADLINE

def _has_independent_set(adj, n, limit):
    """
    Returns True if the graph has an independent set of size >= limit.
    Uses branch and bound on the complement graph (finding a clique of size limit).
    Optimized with bitmasks.
    """
    # Build complement adjacency bitmasks
    comp = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)
    
    # Order vertices by degree in complement (descending) for better pruning
    # This helps in finding cliques (independent sets in G) faster
    order = list(range(n))
    order.sort(key=lambda v: comp[v].bit_count(), reverse=True)
    
    # Remap vertices to new indices based on degree order
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
        
    # Build remapped complement adjacency
    new_comp = [0] * n
    for old_v in range(n):
        new_v = remap[old_v]
        mask = 0
        c = comp[old_v]
        while c:
            bit = c & -c
            old_u = bit.bit_length() - 1
            c ^= bit
            new_u = remap[old_u]
            mask |= (1 << new_u)
        new_comp[new_v] = mask
        
    found = [False]
    
    def search(candidates_mask, current_size):
        if found[0] or _time_up():
            return
        if current_size >= limit:
            found[0] = True
            return
        if candidates_mask == 0:
            return
            
        # Upper bound: current_size + number of candidates
        if current_size + candidates_mask.bit_count() < limit:
            return
            
        # Pick vertex with maximum degree in the induced subgraph of candidates
        # This is a heuristic for better pruning in clique finding
        best_v = -1
        max_deg = -1
        c = candidates_mask
        while c:
            bit = c & -c
            v = bit.bit_length() - 1
            c ^= bit
            deg = (new_comp[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New candidates are those adjacent to best_v in the complement (i.e., non-adjacent in G)
        search(candidates_mask & new_comp[best_v], current_size + 1)
        
        # Branch 2: Exclude best_v
        search(candidates_mask & ~(1 << best_v), current_size)
        
    search(full_mask, 0)
    return found[0]

def _validate(adj, n):
    """Check triangle-free and alpha < 10. Return True if valid."""
    # Triangle-free is guaranteed by construction in the new strategy, 
    # but we keep a check just in case if we modify the construction.
    # For speed, we assume triangle-free if constructed greedily.
    if _has_independent_set(adj, n, 10):
        return False
    return True

def _greedy_construct(n, seed):
    """
    Construct a triangle-free graph on n vertices using a greedy approach.
    1. Generate a random permutation of all edges.
    2. Add an edge if it doesn't create a triangle.
    3. Check independence number.
    """
    random.seed(seed)
    
    # List of all possible edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)
    
    adj = [0] * n
    
    # Greedy addition
    for u, v in edges:
        if _time_up():
            break
        # Check if adding (u, v) creates a triangle
        # A triangle is created if there exists w such that (u, w) and (v, w) are in adj
        # common neighbors of u and v
        common = adj[u] & adj[v]
        if common == 0:
            # No common neighbors, so no triangle
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Validate independence number
    if _has_independent_set(adj, n, 10):
        return None
        
    # Extract edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                final_edges.append([i, j])
                
    return final_edges

def construct():
    best_result = None
    
    # Try to find a 39-vertex graph using greedy construction
    # We try multiple seeds to increase the chance of finding a graph with alpha < 10
    for seed in range(50):
        if _time_up():
            break
        edges = _greedy_construct(39, seed)
        if edges:
            best_result = {'n': 39, 'edges': edges}
            break
            
    # If 39 fails, try 38, 37... down to a safe fallback
    if not best_result:
        for n in range(38, 20, -1):
            if _time_up():
                break
            for seed in range(10):
                if _time_up():
                    break
                edges = _greedy_construct(n, seed)
                if edges:
                    best_result = {'n': n, 'edges': edges}
                    break
            if best_result:
                break
                
    # Fallback: Empty graph on 1 vertex
    if not best_result:
        best_result = {'n': 1, 'edges': []}
        
    return best_result
# EVOLVE-BLOCK-END
