import time
import random
import numpy as np

def _has_non_edge_in_mask(adj, mask, n):
    """Check if induced subgraph on mask has a non-edge."""
    m = mask
    while m:
        lsb = m & -m
        c = lsb.bit_length() - 1
        m ^= lsb
        # d in m (which are all > c)
        if m & ~adj[c]:
            return True
    return False

def _has_k222(adj, n):
    """
    Check if graph represented by bitmask adjacency 'adj' contains K2,2,2.
    adj[i] is an integer bitmask of neighbors of i.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            
            mask_a = C
            while mask_a:
                lsb_a = mask_a & -mask_a
                a = lsb_a.bit_length() - 1
                mask_a ^= lsb_a
                
                mask_b = C & (0xFFFFFFFF << (a + 1))
                while mask_b:
                    lsb_b = mask_b & -mask_b
                    b = lsb_b.bit_length() - 1
                    mask_b ^= lsb_b
                    
                    if adj[a] & (1 << b):
                        continue
                    
                    D = (adj[a] & adj[b]) & C
                    D &= ~(1 << a)
                    D &= ~(1 << b)
                    if D.bit_count() < 2:
                        continue
                    
                    if _has_non_edge_in_mask(adj, D, n):
                        return True
    return False

def _count_k222(adj, n):
    """
    Count number of K2,2,2 subgraphs.
    """
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            
            mask_a = C
            while mask_a:
                lsb_a = mask_a & -mask_a
                a = lsb_a.bit_length() - 1
                mask_a ^= lsb_a
                
                mask_b = C & (0xFFFFFFFF << (a + 1))
                while mask_b:
                    lsb_b = mask_b & -mask_b
                    b = lsb_b.bit_length() - 1
                    mask_b ^= lsb_b
                    
                    if adj[a] & (1 << b):
                        continue
                    
                    D = (adj[a] & adj[b]) & C
                    D &= ~(1 << a)
                    D &= ~(1 << b)
                    if D.bit_count() < 2:
                        continue
                    
                    # Count non-edges in D
                    # Iterate c in D
                    m = D
                    while m:
                        lsb_c = m & -m
                        c = lsb_c.bit_length() - 1
                        m ^= lsb_c
                        # d in m (d > c)
                        # non_edges = m & ~adj[c]
                        count += (m & ~adj[c]).bit_count()
    return count

def _build_adj_masks(colors, n):
    """Build adjacency bitmasks for both colors."""
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    return adj1, adj0

def _get_edge_idx(i, j, n):
    """Get index in colors list for edge (i,j)."""
    # i < j
    # idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    # Simpler: precompute or use formula
    # Number of edges before i is i*(n-1) - i*(i+1)/2
    # Number of edges from i to j is j - i - 1
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _local_search(colors, n, time_limit=100):
    """
    Local search to minimize violations.
    Returns True if valid coloring found.
    Modifies colors in place.
    """
    start_time = time.time()
    
    # Build initial adj masks
    adj1, adj0 = _build_adj_masks(colors, n)
    
    # Initial count
    cur_v1 = _count_k222(adj1, n)
    cur_v0 = _count_k222(adj0, n)
    cur_total = cur_v1 + cur_v0
    
    if cur_total == 0:
        return True
    
    best_total = cur_total
    best_colors = colors[:]
    
    n_edges = n * (n - 1) // 2
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cool_rate = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
        
        # Pick random edge to flip
        idx = random.randint(0, n_edges - 1)
        
        # Find i, j for idx
        # We can precompute a list of (i,j) for each idx
        # Or solve quadratic.
        # For speed, precompute edge_list
        # But we don't have it here. Let's create a helper or precompute outside.
        # For now, let's pass edge_list as arg or compute.
        # Computing i,j from idx:
        # idx = i*(n-1) - i*(i+1)/2 + (j-i-1)
        # This is annoying. Let's precompute edge_list in construct and pass it.
        
        # Let's assume we have edge_list
        # I will refactor _local_search to take edge_list
        
        # ... (Refactoring needed)
        
        # For now, just use random flip and recalc
        # This is slow.
        
        # Let's optimize by passing edge_list
        pass

def construct():
    deadline = time.time() + 140 # Safety margin
    
    # Precompute edge lists for n=29 and n=30
    def get_edge_list(n):
        el = []
        for i in range(n):
            for j in range(i + 1, n):
                el.append((i, j))
        return el
    
    edge_list_29 = get_edge_list(29)
    edge_list_30 = get_edge_list(30)
    
    # Try n=30
    # Initialize with Paley(29) + 1
    colors_29 = _paley(29, 29)
    if colors_29:
        # Extend to 30
        # New vertex 29 connects to 0..28
        # We can initialize these edges randomly or based on Paley extension?
        # Paley(29) is defined on Z_29. Vertex 29 doesn't exist in Z_29.
        # Let's just set them randomly.
        colors_30 = colors_29[:]
        for i in range(29):
            colors_30.append(random.randint(0, 1))
        
        # Run local search for n=30
        # We need a fast local search
        
        # Let's implement a specific local search for n=30
        
        start_search = time.time()
        remaining = deadline - start_search - 5 # 5s buffer
        
        # Local search loop
        # ...
        
        # If we find valid, return
        # else fallback to 29
        
    # Fallback n=29
    colors_29 = _paley(29, 29)
    if colors_29:
        adj1, adj0 = _build_adj_masks(colors_29, 29)
        if not _has_k222(adj1, 29) and not _has_k222(adj0, 29):
            return {'n': 29, 'colors': colors_29}
            
    return {'n': 1, 'colors': []}
