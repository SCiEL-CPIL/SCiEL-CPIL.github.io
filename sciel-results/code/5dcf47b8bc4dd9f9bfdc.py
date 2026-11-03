import random
from collections import deque
import numpy as np
from scipy import sparse

def construct(k, g):
    # 1. Determine search range for n
    # Moore Bound M
    # M = 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i for even g?
    # For odd g, the bound is different.
    # k=5, g=9 (odd)
    # M = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i + ... ?
    # Let's just use a safe lower bound.
    # For (5,9), M = 428.
    
    def calc_moore(k, g):
        # Lower bound on vertices
        # 1 + k * (1 + (k-1) + ... + (k-1)^{m}) where m depends on g
        # If g is odd, g=2r+1, max depth r.
        # M = 1 + k * sum_{i=0}^{r-1} (k-1)^i
        r = (g - 1) // 2
        if r < 1: return 1
        total = 1
        cur = 1
        for i in range(r):
            total += cur
            cur *= (k - 1)
        return 1 + k * (total - 1) # Wait, sum includes 1?
        # Sum_{i=0}^{r-1} (k-1)^i = ((k-1)^r - 1) / (k-2)
        # Let's just calculate iteratively
        # Vertices at dist 0: 1
        # Dist 1: k
        # Dist 2: k(k-1)
        # ...
        # Dist r: k(k-1)^{r-1}
        
        total_verts = 1
        for i in range(r):
            if i == 0:
                total_verts += k
            else:
                total_verts += k * (k-1)**(i-1) * (k-1) # k * (k-1)^i
        return total_verts

    M = calc_moore(k, g)
    
    # Start searching from M
    # Random graphs of size M might be too constrained.
    # We'll scan n from M to M * 2.
    
    def gen_random_regular(n, k):
        # Configuration model
        half_edges = [i // k for i in range(n * k)]
        random.shuffle(half_edges)
        
        adj = [[] for _ in range(n)]
        
        for i in range(n * k // 2):
            u = half_edges[2 * i]
            v = half_edges[2 * i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify degree exactly k
        for u in range(n):
            if len(adj[u]) != k:
                return None
            if len(set(adj[u])) != k:
                return None
                
        return adj

    def has_girth_ge(adj, n, k, g):
        # Use sparse matrix powers to detect cycles of length < g
        # Since the graph is simple, we check for closed walks of length L < g
        # that are not back-and-forth (which would be length 2).
        # For simple graphs, A^L diagonal > 0 implies a cycle of length L or divisor.
        # We need to check L = 3, 4, 5, 6, 7, 8 (if g=9).
        
        # Build sparse adjacency matrix
        rows = []
        cols = []
        for u in range(n):
            for v in adj[u]:
                rows.append(u)
                cols.append(v)
        
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        
        # Check for cycles of length L
        # L=3: A^3
        # L=4: A^4
        # ...
        
        # Optimization: Check even and odd separately?
        # Just compute powers up to g-1.
        
        A_pow = A.copy()
        # A^1 is A. No cycles of length 1 in simple graph.
        
        for L in range(3, g):
            A_pow = A_pow @ A
            diag = np.asarray(A_pow.diagonal())
            if np.any(diag > 0):
                return False
        
        return True

    # Search
    # Expand search range to 1116 (known best)
    max_n = max(M * 2, 1116)
    for n in range(M, max_n + 1):
        # Try random instances
        # Reduce attempts if n is large to save time?
        # 120s limit.
        # Sparse matrix mult is fast.
        attempts = 50 if n > 1000 else 100
        for _ in range(attempts):
            adj = gen_random_regular(n, k)
            if adj is None:
                continue
            if has_girth_ge(adj, n, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback: Return seed construction if random fails (unlikely)
    # Re-implement seed logic here as fallback just in case
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
