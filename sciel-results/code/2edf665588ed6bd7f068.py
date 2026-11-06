import time
import random
from collections import defaultdict

def _build_polarity_graph_pg2_5():
    """
    Construct the polarity graph of PG(2,5).
    Vertices: 31 points in PG(2,5).
    Edges: u ~ v if u is on the polar line of v (i.e., dot product is 0).
    """
    # Generate all points in PG(2,5)
    # Represent points as tuples (a,b,c) in F5^3, not all 0, normalized
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero element 1
                # But we need to be careful with the field arithmetic
                # Actually, just generate unique projective points
                # We can do this by iterating and checking if it's a multiple of an existing point
                pass
    
    # Simpler: iterate over all non-zero vectors, divide by first non-zero component
    seen = set()
    pts = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Find first non-zero
                if a != 0:
                    inv_a = pow(a, 3, 5) # 5 is prime, a^(5-2) = a^3
                    na, nb, nc = 1, (b * inv_a) % 5, (c * inv_a) % 5
                elif b != 0:
                    inv_b = pow(b, 3, 5)
                    na, nb, nc = 0, 1, (c * inv_b) % 5
                else:
                    inv_c = pow(c, 3, 5)
                    na, nb, nc = 0, 0, 1
                
                p = (na, nb, nc)
                if p not in seen:
                    seen.add(p)
                    pts.append(p)
    
    n = len(pts)
    # n should be 31
    
    # Build adjacency
    # u ~ v if dot(u, v) == 0 mod 5
    adj = [0] * n
    for i in range(n):
        u = pts[i]
        for j in range(i + 1, n):
            v = pts[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % 5
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    return n, adj, pts

def _find_independent_set(adj, n, k):
    """Find an independent set of size k. Returns list of indices or None."""
    # Simple backtracking
    # Order by degree
    deg = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda x: deg[x])
    
    found = [None]
    
    def bt(idx, current_set, forbidden):
        if found[0]:
            return
        if len(current_set) == k:
            found[0] = current_set[:]
            return
        if idx >= n:
            return
        v = order[idx]
        if forbidden & (1 << v):
            bt(idx + 1, current_set, forbidden)
            return
        # Try including v
        current_set.append(v)
        new_forbidden = forbidden | (1 << v) | adj[v]
        bt(idx + 1, current_set, new_forbidden)
        current_set.pop()
        if not found[0]:
            bt(idx + 1, current_set, forbidden)

    bt(0, [], 0)
    return found[0]

def _verify_c4_free(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            common = adj[u] & adj[v]
            if common & (common - 1):
                return False
    return True

def _verify_no_ind_11(adj, n):
    # Check if there is an independent set of size 11
    # Use backtracking with pruning
    # This might be slow for n=39, but alpha is small so it should prune fast
    # Actually, we can just check max independent set size using a simple greedy + branch and bound
    # Or just rely on the construction logic if we are sure.
    # For safety, let's do a bounded search.
    
    # Order by degree
    deg = [bin(adj[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda x: deg[x])
    
    found = [False]
    
    def bt(idx, size, forbidden):
        if found[0]:
            return
        if size >= 11:
            found[0] = True
            return
        if idx >= n:
            return
        if size + (n - idx) < 11:
            return
        v = order[idx]
        if forbidden & (1 << v):
            bt(idx + 1, size, forbidden)
            return
        # Skip v
        bt(idx + 1, size, forbidden)
        if found[0]:
            return
        # Include v
        new_forbidden = forbidden | (1 << v) | adj[v]
        bt(idx + 1, size + 1, new_forbidden)
        
    bt(0, 0, 0)
    return not found[0]

def construct():
    deadline = time.time() + 110.0
    
    # 1. Build PG(2,5) Polarity Graph (31 vertices)
    n_base, adj_base, pts = _build_polarity_graph_pg2_5()
    
    if n_base != 31:
        # Fallback to 30 triangles if something went wrong
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
            adj_fallback[u] |= (1 << v) | (1 << w)
            adj_fallback[v] |= (1 << u) | (1 << w)
            adj_fallback[w] |= (1 << u) | (1 << v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    # 2. Find an independent set of size 5 in the base graph
    indep_set = _find_independent_set(adj_base, n_base, 5)
    if indep_set is None:
        # Try size 4
        indep_set = _find_independent_set(adj_base, n_base, 4)
        if indep_set is None:
             # Fallback
             n_fallback = 30
             adj_fallback = [0] * n_fallback
             for i in range(10):
                 u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
                 adj_fallback[u] |= (1 << v) | (1 << w)
                 adj_fallback[v] |= (1 << u) | (1 << w)
                 adj_fallback[w] |= (1 << u) | (1 << v)
             edges = []
             for u in range(n_fallback):
                 for v in range(u + 1, n_fallback):
                     if adj_fallback[u] & (1 << v):
                         edges.append([u, v])
             return {'n': n_fallback, 'edges': edges}
             
    # 3. Extend to 39 vertices
    # Add 8 new vertices: 31..38
    n_new = 8
    n_total = n_base + n_new
    
    adj = [0] * n_total
    
    # Copy base graph
    for i in range(n_base):
        adj[i] = adj_base[i]
        
    # Connect new vertices to each other (clique)
    for i in range(n_base, n_total):
        for j in range(i + 1, n_total):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Connect new vertices to the independent set in base
    for i in range(n_base, n_total):
        for v in indep_set:
            adj[i] |= (1 << v)
            adj[v] |= (1 << i)
            
    # 4. Verify
    if not _verify_c4_free(adj, n_total):
        # Fallback to base graph
        edges = []
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if adj_base[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_base, 'edges': edges}
        
    if not _verify_no_ind_11(adj, n_total):
        # Fallback to base graph
        edges = []
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if adj_base[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_base, 'edges': edges}
        
    # 5. Build edge list
    edges = []
    for u in range(n_total):
        for v in range(u + 1, n_total):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return {'n': n_total, 'edges': edges}
