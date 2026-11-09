import time
import random
import sys

def _alpha_bnb_fast(adj, n, limit=10):
    """
    Fast branch-and-bound: returns True if alpha >= limit, False if alpha < limit.
    """
    best = [0]
    
    def bnb(cand, sz):
        if sz >= limit:
            best[0] = limit
            return True
        if sz + cand.bit_count() < limit:
            return False
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return False
        
        # Pick vertex with max degree in cand for better pruning
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return False

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        if bnb(cand & ~v_bit & ~adj[best_v], sz + 1):
            return True
        
        # Branch 2: Exclude best_v
        if bnb(cand & ~v_bit, sz):
            return True
        
        if sz > best[0]:
            best[0] = sz
        return False

    bnb((1 << n) - 1, 0)
    return best[0]

def _is_triangle_free_fast(adj, n):
    """Check triangle-free using bitmask operations."""
    for u in range(n):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            if u < v:
                if (adj[u] & adj[v]) != 0:
                    return False
            v_bits ^= lsb
    return True

def _find_independent_set(adj, n, limit=10):
    """Find an independent set of size 'limit' if it exists."""
    found = [None]
    
    def bnb(cand, sz, path):
        if found[0] is not None:
            return
        if sz >= limit:
            found[0] = list(path)
            return
        if sz + cand.bit_count() < limit:
            return
        if cand == 0:
            return
        
        # Pick vertex with min degree in cand (more likely to be in IS)
        best_v = -1
        best_deg = n + 1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        path.append(best_v)
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1, path)
        path.pop()
        
        if found[0] is not None:
            return
        bnb(cand & ~v_bit, sz, path)
        
    bnb((1 << n) - 1, 0, [])
    return found[0]

def _local_search(n, seed, time_limit):
    """
    Local search to find a triangle-free graph with alpha < 10.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize with a random triangle-free graph
    # Strategy: Start with bipartite, then add edges carefully?
    # Or just random edges that don't form triangles.
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Check if already valid
    alpha = _alpha_bnb_fast(adj, n, limit=10)
    if alpha < 10:
        return adj
        
    # Local Search Loop
    while time.time() - start_time < time_limit:
        alpha = _alpha_bnb_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
            
        is_set = _find_independent_set(adj, n, limit=10)
        if is_set is None:
            return adj # Should be caught above
            
        # Try to add an edge within is_set
        modified = False
        # Shuffle to avoid bias
        is_set_copy = is_set[:]
        rng.shuffle(is_set_copy)
        
        for i in range(len(is_set_copy)):
            for j in range(i + 1, len(is_set_copy)):
                u, v = is_set_copy[i], is_set_copy[j]
                if (adj[u] & (1 << v)) == 0: # Not already connected
                    if (adj[u] & adj[v]) == 0: # Adding edge doesn't create triangle
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        modified = True
                        break
            if modified:
                break
                
        if not modified:
            # Stuck: Restart with a new random graph
            adj = [0] * n
            edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            # Quick check after restart
            if _alpha_bnb_fast(adj, n, limit=10) < 10:
                return adj

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Andrasfai graph And(3) on 26 vertices
    # Actually, let's use a known valid graph if we can construct one quickly.
    # But we will rely on local search.
    
    best_result = None
    
    # Try n=39 first (known to be possible)
    for seed in range(100):
        if time.time() > deadline:
            break
        adj = _local_search(39, seed, time_limit=0.5)
        if adj is not None:
            edges = []
            for u in range(39):
                v_bits = adj[u]
                while v_bits:
                    lsb = v_bits & -v_bits
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    v_bits ^= lsb
            best_result = {'n': 39, 'edges': edges}
            break
            
    if best_result:
        # If we have time, try n=40
        if time.time() < deadline - 10:
            for seed in range(100):
                if time.time() > deadline - 5:
                    break
                adj = _local_search(40, seed, time_limit=0.5)
                if adj is not None:
                    edges = []
                    for u in range(40):
                        v_bits = adj[u]
                        while v_bits:
                            lsb = v_bits & -v_bits
                            v = lsb.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            v_bits ^= lsb
                    best_result = {'n': 40, 'edges': edges}
                    break
    
    if best_result:
        return best_result
        
    # Final fallback: Return a valid smaller graph
    # Construct Andrásfai graph And(3) on 26 vertices
    n_fallback = 26
    edges_fallback = []
    for i in range(n_fallback):
        for j in range(i + 1, n_fallback):
            if (j - i) % n_fallback % 3 == 1:
                edges_fallback.append([i, j])
                
    return {'n': n_fallback, 'edges': edges_fallback}
