import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmask list adj on n vertices contains K2,2,2.
    Returns True if found.
    """
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a)
                    cn_mask &= ~(1 << b)
                    cn_mask &= ~(1 << c)
                    cn_mask &= ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        return True
    return False

def _is_valid_coloring(adj, n):
    """Check if neither color class contains K2,2,2."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _build_paley29():
    """Build Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _try_extend_to_30(base_adj, deadline):
    """
    Try to extend the 29-vertex base to 30 vertices using random restarts and local search.
    Returns a list of 29 colors (0 or 1) for edges (29, i) if a valid coloring is found, else None.
    """
    n_base = 29
    n_target = 30
    random.seed(42)
    
    # Precompute base adjacency for speed
    base_adj_list = base_adj[:]
    
    for restart in range(200): # Limit restarts to fit in time
        if time.time() > deadline:
            break
            
        x = [random.randint(0, 1) for _ in range(n_base)]
        
        # Check if this random assignment works
        adj = [0] * n_target
        for i in range(n_base):
            adj[i] = base_adj_list[i]
        for i in range(n_base):
            if x[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
                
        if _is_valid_coloring(adj, n_target):
            return x[:]
            
        # Local search: flip bits to try to fix
        for step in range(50):
            if time.time() > deadline:
                break
            i = random.randint(0, n_base - 1)
            x[i] = 1 - x[i]
            
            # Rebuild adj
            adj = [0] * n_target
            for i2 in range(n_base):
                adj[i2] = base_adj_list[i2]
            for i2 in range(n_base):
                if x[i2] == 1:
                    adj[29] |= (1 << i2)
                    adj[i2] |= (1 << 29)
                    
            if _is_valid_coloring(adj, n_target):
                return x[:]
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Build and Validate n=29 base
    base_adj = _build_paley29()
    
    # 2. Try to extend to n=30
    # Use a shorter deadline for the search to ensure we have time to format and return
    search_deadline = start_time + 110
    
    result_edges = _try_extend_to_30(base_adj, search_deadline)
    
    if result_edges is not None:
        # Build the 30-vertex graph
        n = 30
        adj = [0] * n
        for i in range(29):
            adj[i] = base_adj[i]
        for i in range(29):
            if result_edges[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        # Final verification before returning
        if _is_valid_coloring(adj, n):
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(1 if (adj[i] >> j) & 1 else 0)
            return {'n': n, 'colors': colors}
            
    # 3. Fallback: Return n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(1 if (base_adj[i] >> j) & 1 else 0)
            
    return {'n': n, 'colors': colors}
