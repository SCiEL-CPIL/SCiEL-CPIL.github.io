import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if the graph defined by adjacency list of bitmasks 'adj' on n vertices 
    contains K2,2,2 as a subgraph. Returns True if found.
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

def _find_violation_edges(adj, n, max_edges=20):
    """
    Find up to max_edges edges that are part of a K2,2,2 in either color class.
    Returns list of (u, v) tuples.
    """
    violations = []
    full = (1 << n) - 1
    
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
                        violations.append((u, v))
                        if len(violations) >= max_edges:
                            return violations
    
    # Check color 1 (complement)
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
                        violations.append((u, v))
                        if len(violations) >= max_edges:
                            return violations
    
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Try to construct a valid coloring for n=30 using simulated annealing
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

    # Simulated Annealing
    best_adj = None
    best_n = 29
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling = 0.999
    
    # Main search loop with simulated annealing
    T = T_start
    for iteration in range(100000):
        if time.time() > deadline - 5:
            break
            
        violations = _find_violation_edges(adj, n, max_edges=10)
        if not violations:
            # Valid coloring found for n=30
            best_adj = adj[:]
            best_n = n
            break
        
        # Pick a random violating edge to flip
        u, v = random.choice(violations)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check if the flip improved things (fewer violations)
        new_violations = _find_violation_edges(adj, n, max_edges=10)
        if len(new_violations) < len(violations):
            # Improvement: always accept
            pass
        else:
            # Worsening or same: accept with probability exp(-delta/T)
            delta = len(new_violations) - len(violations)
            if delta > 0 and random.random() < (2.718281828 ** (-delta / max(T, 0.001))):
                pass  # Accept worsening
            else:
                # Reject: undo the flip
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        T *= cooling
    
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
