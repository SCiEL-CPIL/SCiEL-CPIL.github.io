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

def _find_violation_edges(adj, n):
    """
    Find all edges that participate in at least one K2,2,2 violation in either color class.
    Returns a set of (u,v) pairs where flipping the edge color might help.
    """
    violating = set()
    
    # Check color 0 class (edges in adj)
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
                        # Found a violation involving edge (u,v)
                        violating.add((u, v))
                        # Also mark edges within the violation structure
                        for c in range(n):
                            if (common_b1_b2 >> c) & 1:
                                violating.add((b1, c) if b1 < c else (c, b1))
                                violating.add((b2, c) if b2 < c else (c, b2))
    
    # Check color 1 class (complement edges)
    full = (1 << n) - 1
    comp = [0] * n
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
                        violating.add((u, v))
                        for c in range(n):
                            if (common_b1_b2 >> c) & 1:
                                violating.add((b1, c) if b1 < c else (c, b1))
                                violating.add((b2, c) if b2 < c else (c, b2))
    
    return violating

def _count_violations(adj, n):
    """Count total number of K2,2,2 violations in both color classes."""
    count = 0
    
    # Color 0 class
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
                        count += 1
    
    # Color 1 class
    full = (1 << n) - 1
    comp = [0] * n
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

def _flip_edge(adj, n, u, v):
    """Flip the color of edge (u,v)."""
    if (adj[u] >> v) & 1:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    else:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

def _paley_29():
    """Construct Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Start with Paley-29
    adj, n = _paley_29()
    
    # Verify it's valid
    if _count_violations(adj, n) == 0:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        best_result = {'n': n, 'colors': colors}
    else:
        # Shouldn't happen, but fallback
        best_result = {'n': 1, 'colors': []}
    
    # Try to extend to 30 vertices
    # Add vertex 29 with initial coloring based on majority rule
    n_new = 30
    adj_new = adj[:]
    adj_new.append(0)
    
    # For each edge (29, i), choose color to minimize violations
    # Start with all edges to vertex 29 as color 0 (in adj)
    for i in range(29):
        adj_new[29] |= (1 << i)
        adj_new[i] |= (1 << 29)
    
    # Local search: try to reduce violations
    current_violations = _count_violations(adj_new, n_new)
    best_adj = adj_new[:]
    best_violations = current_violations
    
    temperature = 10.0
    cooling = 0.999
    
    # Simulated annealing
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, n_new - 1)
        v = random.randint(0, n_new - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_violations = current_violations
        _flip_edge(adj_new, n_new, u, v)
        new_violations = _count_violations(adj_new, n_new)
        
        delta = new_violations - old_violations
        if delta <= 0 or random.random() < pow(2.718281828, -delta / temperature):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj_new[:]
                if best_violations == 0:
                    break
        else:
            _flip_edge(adj_new, n_new, u, v)
        
        temperature *= cooling
    
    # Check if we found a valid 30-vertex coloring
    if best_violations == 0:
        colors = []
        for i in range(n_new):
            for j in range(i + 1, n_new):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n_new, 'colors': colors}
    
    # If we couldn't extend to 30, return the best 29-vertex result
    # Check if best_adj restricted to first 29 vertices is valid
    adj_29 = best_adj[:29]
    # Need to rebuild adj_29 properly since best_adj might have modified edges
    adj_29 = [0] * 29
    for i in range(29):
        for j in range(i + 1, 29):
            if (best_adj[i] >> j) & 1:
                adj_29[i] |= (1 << j)
                adj_29[j] |= (1 << i)
    
    if _count_violations(adj_29, 29) == 0:
        colors = []
        for i in range(29):
            for j in range(i + 1, 29):
                if (adj_29[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': 29, 'colors': colors}
    
    # Fallback to original Paley-29
    return best_result
