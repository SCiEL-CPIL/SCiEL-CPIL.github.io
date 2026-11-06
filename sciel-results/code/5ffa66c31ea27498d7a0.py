import time
import random
import math

_DEADLINE = None

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _build_35_vertex_graph():
    """Construct the known 35-vertex (4,6)-Ramsey graph."""
    n = 35
    adj = [0] * n
    S7 = {1, 2, 4}
    S5 = {1, 2, 3, 4}
    
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1 + 1, 7):
                for j2 in range(5):
                    v2 = i2 * 5 + j2
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj

def _init_36_graph(adj35, density=0.5, seed=0):
    """Initialize a 36-vertex graph from a 35-vertex graph."""
    n = 36
    adj = adj35[:] + [0]
    random.seed(seed)
    for i in range(35):
        if random.random() < density:
            adj[35] |= (1 << i)
            adj[i] |= (1 << 35)
    return adj

def _repair_and_search(adj, n, deadline):
    """
    Perform local search/repair on the graph.
    Returns True if a valid graph is found (and modifies adj in place).
    """
    # Track state to detect stagnation
    last_valid_check = time.time()
    stagnation_counter = 0
    max_stagnation = 5000 # Steps before a kick
    
    while time.time() < deadline:
        # 1. Check for K4
        k4 = _k4_found(adj, n)
        if k4:
            # Remove a random edge from the K4
            a, b, c, d = k4
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            stagnation_counter += 1
            continue
        
        # 2. Check for Ind-6
        ind6 = _ind6_found(adj, n)
        if ind6:
            # Add a random edge to the Ind-6
            a, b, c, d, e, f = ind6
            pairs = [(a,b), (a,c), (a,d), (a,e), (a,f),
                     (b,c), (b,d), (b,e), (b,f),
                     (c,d), (c,e), (c,f),
                     (d,e), (d,f),
                     (e,f)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            stagnation_counter += 1
            continue
        
        # 3. Valid!
        return True

        # Stagnation check (if we loop too long without finding violation? 
        # Actually, if we find violations, we fix them. 
        # Stagnation happens if we oscillate or if the repair moves are ineffective.
        # A simple "kick" helps escape local minima.
        
        # We can't easily detect "oscillation" without state history.
        # Let's do periodic kicks.
        if stagnation_counter > max_stagnation:
            # Kick: Flip 5 random edges
            for _ in range(5):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            stagnation_counter = 0
            max_stagnation = int(max_stagnation * 1.2) # Increase patience

    return False

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 110 # 10s margin

    # Get the 35-vertex base
    adj35 = _build_35_vertex_graph()
    
    # Verify base is valid (sanity check)
    if _k4_found(adj35, 35) is not None or _ind6_found(adj35, 35) is not None:
        # Fallback to random search if base is broken (should not happen)
        pass

    # Try to extend to 36
    # Multiple attempts with different initializations and seeds
    attempts = 0
    max_attempts = 100
    
    # Try different densities for the new vertex
    densities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.45, 0.55, 0.35, 0.65, 0.5]
    
    for density in densities:
        if time.time() > _DEADLINE:
            break
            
        # Try multiple random seeds for this density
        for seed_offset in range(10):
            if time.time() > _DEADLINE:
                break
                
            seed = 1000 * density + seed_offset
            adj36 = _init_36_graph(adj35, density, seed)
            
            # Run repair search
            if _repair_and_search(adj36, 36, _DEADLINE):
                return {'n': 36, 'edges': _edges_from_adj(adj36, 36)}
                
    # If 36 fails, return 35
    if time.time() < _DEADLINE:
        return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
        
    # Emergency fallback
    return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
