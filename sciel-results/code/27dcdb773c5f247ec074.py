import time
import random
from itertools import combinations

def _has_k222(adj, n):
    """
    Check if the graph defined by adjacency list of bitmasks 'adj' on n vertices 
    contains K2,2,2 as a subgraph.
    adj[i] is a bitmask of neighbors of i.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222(comp, n):
        return False
    return True

def _count_violations(adj, n):
    """Count the number of K2,2,2 subgraphs in both color classes."""
    count = 0
    if _has_k222(adj, n):
        # For optimization, we might want the actual count or just existence.
        # For local search, existence is enough to trigger a flip.
        # But counting helps to choose the best flip.
        # Let's stick to existence for speed, or a simple count.
        pass
    return 0 # Placeholder, we will use _is_valid or a specific violation finder

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Try to construct a valid coloring for n=30 using local search
    # Start with Paley(29) + 1 random vertex
    n_target = 30
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = n_target
    adj = [0] * n
    
    # Initialize first 29 vertices with Paley graph
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initialize vertex 29 (index 29) randomly
    for i in range(q):
        if random.random() < 0.5:
            adj[i] |= (1 << 29)
            adj[29] |= (1 << i)

    # Local Search
    # We need a function to find a violating edge or just check validity.
    # To speed up, we can check validity. If invalid, pick a random edge and flip.
    # Better: Find a specific K2,2,2 and flip one of its edges.
    
    def _find_violation(adj, n):
        """Returns a tuple (u, v) of an edge that is part of a K2,2,2, or None."""
        # Check color 0
        if _has_k222(adj, n):
            # Find one
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    common_mask = adj[u] & adj[v]
                    common_mask &= ~(1 << u)
                    common_mask &= ~(1 << v)
                    cn_list = []
                    m = common_mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cn_list.append(idx)
                        m ^= lsb
                    L = len(cn_list)
                    for i in range(L):
                        b1 = cn_list[i]
                        for j in range(i + 1, L):
                            b2 = cn_list[j]
                            common_b1_b2 = adj[b1] & adj[b2] & common_mask
                            if common_b1_b2.bit_count() >= 2:
                                # Found a K2,2,2 involving edge (u,v)
                                return (u, v)
            return (0, 1) # Fallback, should not happen if _has_k222 is True
            
        # Check color 1 (complement)
        comp = [0] * n
        full = (1 << n) - 1
        for i in range(n):
            comp[i] = full ^ adj[i] ^ (1 << i)
            
        if _has_k222(comp, n):
            for u in range(n):
                for v in range(u + 1, n):
                    if not (comp[u] >> v) & 1:
                        continue
                    common_mask = comp[u] & comp[v]
                    common_mask &= ~(1 << u)
                    common_mask &= ~(1 << v)
                    cn_list = []
                    m = common_mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cn_list.append(idx)
                        m ^= lsb
                    L = len(cn_list)
                    for i in range(L):
                        b1 = cn_list[i]
                        for j in range(i + 1, L):
                            b2 = cn_list[j]
                            common_b1_b2 = comp[b1] & comp[b2] & common_mask
                            if common_b1_b2.bit_count() >= 2:
                                # Edge (u,v) is 0 in comp, so 1 in adj.
                                return (u, v)
            return (0, 1)
            
        return None

    # Main search loop
    best_adj = None
    best_n = 29 # Fallback
    
    # Try to improve n=30
    for _ in range(50000): # Max iterations
        if time.time() > deadline - 5:
            break
            
        violation = _find_violation(adj, n)
        if violation is None:
            # Valid coloring found for n=30
            best_adj = adj[:]
            best_n = n
            break
        
        u, v = violation
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    if best_n == 30:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
        
    # Fallback to n=29
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
