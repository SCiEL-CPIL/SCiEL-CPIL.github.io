import time
import random

def _has_k222_fast(adj, n):
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

def _find_violation_edges(adj, n):
    """
    Returns a list of edges (u, v) that are part of a K2,2,2 subgraph in the given graph.
    If no K2,2,2 exists, returns empty list.
    """
    edges = set()
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
                        # Found a K2,2,2. The edges involved are:
                        # (u,v), (b1,c), (b2,c) for c in common_b1_b2
                        # (b1,b2) is not necessarily an edge, but (b1,c) and (b2,c) are.
                        # We want to flip edges that are part of the subgraph.
                        # The core edges forming the K2,2,2 structure are:
                        # (u,v) [part A]
                        # (u,b1), (u,b2), (v,b1), (v,b2) [A-B edges]
                        # (b1,c), (b2,c) for c in C [B-C edges]
                        # (u,c), (v,c) for c in C [A-C edges]
                        
                        # Let's add all edges incident to the 6 vertices that are present in the graph.
                        # Actually, just adding (u,v) is a good candidate for flipping.
                        # Also (b1, c) and (b2, c) are good candidates.
                        # Let's add (u,v) and a few others.
                        edges.add((u, v))
                        
                        # Add B-C edges
                        m2 = common_b1_b2
                        while m2:
                            lsb = m2 & (-m2)
                            c_idx = lsb.bit_length() - 1
                            edges.add((b1, c_idx) if b1 < c_idx else (c_idx, b1))
                            edges.add((b2, c_idx) if b2 < c_idx else (c_idx, b2))
                            m2 ^= lsb
                        
                        # Add A-B edges
                        edges.add((u, b1) if u < b1 else (b1, u))
                        edges.add((u, b2) if u < b2 else (b2, u))
                        edges.add((v, b1) if v < b1 else (b1, v))
                        edges.add((v, b2) if v < b2 else (b2, v))
                        
                        # Add A-C edges
                        m3 = common_b1_b2
                        while m3:
                            lsb = m3 & (-m3)
                            c_idx = lsb.bit_length() - 1
                            edges.add((u, c_idx) if u < c_idx else (c_idx, u))
                            edges.add((v, c_idx) if v < c_idx else (c_idx, v))
                            m3 ^= lsb
    return list(edges)

def _count_violations_fast(adj, n):
    """
    Count the number of K2,2,2 subgraphs in the given graph.
    """
    count = 0
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
                    bc = common_b1_b2.bit_count()
                    if bc >= 2:
                        # Number of ways to choose 2 from bc
                        count += bc * (bc - 1) // 2
    return count

def _count_violations_both(adj, n):
    """
    Count total violations in both color classes.
    """
    c0 = _count_violations_fast(adj, n)
    
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    
    c1 = _count_violations_fast(comp, n)
    
    return c0 + c1

def _paley_29():
    """Return Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 29
    best_adj = _paley_29()
    
    # Verify best_adj is valid (should be)
    # c0 = _count_violations_fast(best_adj, 29)
    # full = (1 << 29) - 1
    # comp = [0] * 29
    # for i in range(29):
    #     comp[i] = full ^ best_adj[i] ^ (1 << i)
    # c1 = _count_violations_fast(comp, 29)
    # if c0 > 0 or c1 > 0:
    #     # Fallback to n=17 if 29 is invalid (should not happen)
    #     q = 17
    #     squares = set()
    #     for x in range(1, q):
    #         squares.add(x * x % q)
    #     adj = [0] * q
    #     for i in range(q):
    #         for j in range(i + 1, q):
    #             diff = (j - i) % q
    #             if diff in squares:
    #                 adj[i] |= (1 << j)
    #                 adj[j] |= (1 << i)
    #     best_n = 17
    #     best_adj = adj
    #     # Return immediately as fallback
    #     colors = []
    #     for i in range(best_n):
    #         for j in range(i + 1, best_n):
    #             if (best_adj[i] >> j) & 1:
    #                 colors.append(0)
    #             else:
    #                 colors.append(1)
    #     return {'n': best_n, 'colors': colors}

    # Try to extend to n=30
    n = 30
    adj = [0] * n
    for i in range(29):
        adj[i] = best_adj[i]
    
    # Multiple random initializations
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Reset vertex 29 connections
        for i in range(29):
            adj[i] &= ~(1 << 29)
        adj[29] = 0
        
        # Random initialization for vertex 29
        for i in range(29):
            if random.random() < 0.5:
                adj[i] |= (1 << 29)
                adj[29] |= (1 << i)
        
        # Local Search
        # We use a hybrid of greedy and simulated annealing
        # Start with greedy improvements
        
        current_violations = _count_violations_both(adj, n)
        
        # Greedy phase
        for g_iter in range(1000):
            if time.time() > deadline - 5:
                break
            
            if current_violations == 0:
                best_n = n
                best_adj = adj[:]
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (best_adj[i] >> j) & 1:
                            colors.append(0)
                        else:
                            colors.append(1)
                return {'n': n, 'colors': colors}
            
            # Find violation edges
            # To save time, we can just pick a random edge if we don't want to compute the full list
            # But picking from violation edges is smarter.
            # Let's limit the computation.
            
            # Check if color 0 has violation
            v_edges_0 = []
            if _has_k222_fast(adj, n):
                v_edges_0 = _find_violation_edges(adj, n)
            
            v_edges_1 = []
            if not v_edges_0:
                full = (1 << n) - 1
                comp = [0] * n
                for i in range(n):
                    comp[i] = full ^ adj[i] ^ (1 << i)
                if _has_k222_fast(comp, n):
                    v_edges_1 = _find_violation_edges(comp, n)
            
            if not v_edges_0 and not v_edges_1:
                break
            
            # Pick a random edge from the violation set
            if v_edges_0:
                u, v = random.choice(v_edges_0)
            else:
                # v_edges_1 contains edges in comp, which are non-edges in adj
                u, v = random.choice(v_edges_1)
            
            # Flip the edge
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_violations = _count_violations_both(adj, n)
            
            if new_violations < current_violations:
                current_violations = new_violations
            else:
                # Undo
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                break
        
        # Simulated Annealing phase if greedy didn't solve it
        if current_violations > 0:
            T = 1.0
            cooling = 0.9995
            min_T = 0.001
            max_sa_iters = 5000
            
            for sa_iter in range(max_sa_iters):
                if time.time() > deadline - 5:
                    break
                
                if current_violations == 0:
                    best_n = n
                    best_adj = adj[:]
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (best_adj[i] >> j) & 1:
                                colors.append(0)
                            else:
                                colors.append(1)
                    return {'n': n, 'colors': colors}
                
                # Pick random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_violations = _count_violations_both(adj, n)
                delta = new_violations - current_violations
                
                if delta <= 0 or (T > min_T and random.random() < (2.718281828 ** (-delta / max(T, 0.001)))):
                    current_violations = new_violations
                    T *= cooling
                else:
                    # Undo
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    T *= cooling
    
    # Fallback to n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
