import time
import sys
from collections import deque
import numpy as np

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    q = k
    
    # Precompute field operations for efficiency
    if q == 3:
        add_tbl = np.array([[0,1,2],[1,2,0],[2,0,1]], dtype=np.int32)
        sub_tbl = np.array([[0,2,1],[1,0,2],[2,1,0]], dtype=np.int32)
        mul_tbl = np.array([[0,0,0],[0,1,2],[0,2,1]], dtype=np.int32)
    elif q == 4:
        table = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]], dtype=np.int32)
        add_tbl = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]], dtype=np.int32)
        sub_tbl = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]], dtype=np.int32)
        mul_tbl = table
    else:
        add_tbl = np.array([[(i+j)%q for j in range(q)] for i in range(q)], dtype=np.int32)
        sub_tbl = np.array([[(i-j)%q for j in range(q)] for i in range(q)], dtype=np.int32)
        mul_tbl = np.array([[(i*j)%q for j in range(q)] for i in range(q)], dtype=np.int32)

    def build_rule(d):
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        return rule

    def build_graph(d, rule):
        # Precompute rule arrays for faster access
        rule_a = np.zeros(d+1, dtype=np.int32)
        rule_b = np.zeros(d+1, dtype=np.int32)
        for c in range(2, d+1):
            rule_a[c], rule_b[c] = rule[c]
        
        # We need to generate all 2*q^d vertices
        # Side 0: points, Side 1: lines
        # Vertex = (side, x1, x2, ..., xd) where x1 is free, x2..xd are determined
        
        # Total vertices per side: q^d
        n_per_side = q ** d
        n_total = 2 * n_per_side
        
        # Generate all point vertices (side 0)
        # For each point, x1 is free (q choices), x2..xd are computed
        # Actually, in LUW, the vertex is (side, (x0=None, x1, x2, ..., xd))
        # x1 is free, x2..xd are determined by the adjacency rule
        
        # Let's think differently: the LUW graph has vertices as:
        # Points: (p1, p2, ..., pd) in GF(q)^d
        # Lines: (l1, l2, ..., ld) in GF(q)^d
        # Adjacency: point p ~ line l iff for all c=2..d: l[c] - p[c] = l[a_c] * p[b_c]
        
        # So we need to enumerate all q^d points and q^d lines
        # and connect them based on the rule
        
        # Generate all points as a 2D array: shape (q^d, d)
        # Each row is (x1, x2, ..., xd)
        
        # Use np.meshgrid or iterative construction
        # For efficiency, use base-q representation
        
        # Generate all d-tuples over GF(q)
        # This is q^d tuples of length d
        # For q=3, d=11: 177147 tuples of length 11
        
        # Method: use np.unravel_index on range(q^d) with shape (q,)*d
        # But q^d might be large. For q=3, d=11: 177147 - manageable
        
        idx = np.arange(n_per_side, dtype=np.int64)
        # Convert to base-q representation
        coords = np.zeros((n_per_side, d), dtype=np.int32)
        for pos in range(d):
            # pos corresponds to coordinate index d-1-pos (most significant first?)
            # Let's use: idx = x1*q^(d-1) + x2*q^(d-2) + ... + xd
            # Actually, let's use: idx = x1 + x2*q + x3*q^2 + ... + xd*q^(d-1)
            # So coordinate j (0-indexed) has weight q^j
            coords[:, pos] = (idx // (q ** pos)) % q
        
        # Now for each point p = (p1, p2, ..., pd), find all lines l such that
        # l[c] = p[c] + l[a_c] * p[b_c] for c = 2..d
        # l[1] is free (q choices)
        # So for each point, there are q lines connected to it
        
        # Build adjacency: for each point, for each l1 in range(q), compute the line
        # and add edge
        
        # Vectorized approach:
        # For a given point p and l1 value, compute l[2..d]
        # l[c] = p[c] + l[a_c] * p[b_c]
        
        # This is recursive because l[a_c] might depend on l[1] or earlier l values
        # Let's trace through:
        # c=2: a=1, b=1: l[2] = p[2] + l[1]*p[1]
        # c=3: a=2, b=1: l[3] = p[3] + l[2]*p[1]
        # c=4: a=1, b=2: l[4] = p[4] + l[1]*p[2]
        # c=5: a=1, b=3: l[5] = p[5] + l[1]*p[3]
        # c=6: a=5, b=1: l[6] = p[6] + l[5]*p[1]
        # etc.
        
        # So we can compute l[2..d] sequentially given l[1] and p
        
        # For all points and all l1 values, compute the corresponding lines
        # Shape: (n_per_side, q) for l1 values
        # Result: (n_per_side, q, d) array of line coordinates
        
        # This might use too much memory for large d. Let's do it in batches.
        
        # Actually, let's just use the original BFS approach but with optimized neighbor computation
        # using precomputed tables
        
        def neighbours(vertex):
            side, x = vertex
            if side == 0:
                for first in range(q):
                    y = [0]*(d+1)
                    y[1] = first
                    for c in range(2, d+1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add_tbl[x[c]][mul_tbl[y[a]][x[b]]]
                    yield 1, tuple(y)
            else:
                for first in range(q):
                    y = [0]*(d+1)
                    y[1] = first
                    for c in range(2, d+1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub_tbl[x[c]][mul_tbl[x[a]][y[b]]]
                    yield 0, tuple(y)

        start = (0, (0,)*(d+1))
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            
            if time.time() - t0 > deadline - 10.0:
                break

        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return n, edges, adj

    def verify_girth(n, adj, min_girth):
        """
        Check girth >= min_girth using non-backtracking BFS from each vertex.
        """
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Theoretical minimum d for girth g
    # LUW girth: d+5 if d odd, d+6 if d even
    # We need girth >= g
    d_min = 3
    while True:
        girth_d = d_min + 5 if d_min % 2 == 1 else d_min + 6
        if girth_d >= g:
            break
        d_min += 1
    
    # Try increasing d until we find a valid graph or run out of time
    max_d = min(d_min + 2, 15)
    
    for d in range(d_min, max_d + 1):
        if time.time() - t0 > deadline - 5.0:
            break
            
        rule = build_rule(d)
        try:
            n, edges, adj = build_graph(d, rule)
        except MemoryError:
            break
        except Exception:
            continue

        # Check girth
        if verify_girth(n, adj, g):
            return n, edges
        
    # Fallback
    return 0, []

# EVOLVE-BLOCK-END
