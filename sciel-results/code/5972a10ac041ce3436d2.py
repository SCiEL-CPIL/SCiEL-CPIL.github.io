import time
import random

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

def _find_violation_edge(adj, n):
    """
    Returns a tuple (u, v) of an edge that is part of a K2,2,2 in either color class,
    or None if valid.
    """
    # Check color 0 (adj)
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
                        return (u, v)
    
    # Check color 1 (complement)
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
        
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
                        return (u, v)
                        
    return None

def _count_violations(adj, n):
    """
    Count the number of K2,2,2 subgraphs in both color classes.
    This is a proxy for the 'energy' of the configuration.
    """
    count = 0
    
    # Count in adj
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
                        count += common_b1_b2.bit_count() # Rough count, just needs to be correlated
                        # Actually, to be precise about "number of K222", we'd need to count distinct sets.
                        # But for optimization, a weighted count of local violations is often better.
                        # Let's just count 1 per (u,v,b1,b2) pair that has >=2 common neighbors.
                        # This is O(N^4) which is fine for N=30.
    # Re-doing count more efficiently for the surrogate
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
                        count += 1 # Count each (u,v,b1,b2) as a violation indicator
                        
    # Count in comp
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
        
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
                        count += 1
                        
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Construct Paley(29) as baseline
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
        
    n_base = q
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
                
    if _is_valid(adj_base, n_base):
        base_colors = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if (adj_base[i] >> j) & 1:
                    base_colors.append(0)
                else:
                    base_colors.append(1)
        # Save baseline
        best_adj = adj_base[:]
        best_n = n_base
    else:
        # Fallback to smaller if 29 fails (shouldn't happen)
        q = 17
        squares = set()
        for x in range(1, q):
            squares.add(x * x % q)
        n_base = q
        adj_base = [0] * n_base
        for i in range(n_base):
            for j in range(i + 1, n_base):
                diff = (j - i) % q
                if diff in squares:
                    adj_base[i] |= (1 << j)
                    adj_base[j] |= (1 << i)
        best_adj = adj_base[:]
        best_n = n_base

    # 2. Attempt to extend to n=30 using Simulated Annealing
    # Start with Paley(29) + 1 random vertex
    n_target = 30
    adj = [0] * n_target
    # Copy base
    for i in range(n_base):
        adj[i] = adj_base[i]
    
    # Initialize new vertex 29
    for i in range(n_base):
        if random.random() < 0.5:
            adj[i] |= (1 << 29)
            adj[29] |= (1 << i)
            
    current_violations = _count_violations(adj, n_target)
    best_adj_30 = None
    best_violations_30 = current_violations
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling = 0.95
    
    while time.time() < deadline - 2.0:
        # Find a random edge to flip
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_violations = _count_violations(adj, n_target)
        delta = new_violations - current_violations
        
        if delta < 0 or (T > 0 and random.random() < pow(2.718281828, -delta / T)):
            # Accept
            current_violations = new_violations
            if new_violations < best_violations_30:
                best_violations_30 = new_violations
                best_adj_30 = adj[:]
                
            if new_violations == 0:
                # Found a valid coloring!
                best_adj = best_adj_30
                best_n = n_target
                break
        else:
            # Reject: revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        T *= cooling
        if T < 0.01:
            # Reheat occasionally or just stop SA and do greedy
            # Let's do a simple reheat to escape local minima
            T = 10.0
            
    # If we found a valid 30, return it
    if best_n == 30:
        colors = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': best_n, 'colors': colors}
        
    # If SA didn't find 30, maybe it found a better 29? Unlikely to improve 29 since 29 is fixed.
    # But we should check if our 29 is still valid (it is, we didn't touch it unless we extended).
    
    # Return the best valid configuration found (which is the base 29)
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': best_n, 'colors': colors}
