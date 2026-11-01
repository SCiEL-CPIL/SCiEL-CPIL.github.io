import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Try Norm Graph NG(4) first if k=5
    if k == 5:
        n_ng, edges_ng = _construct_norm_graph(4)
        if n_ng > 0:
            # Check girth
            girth_ng = _compute_girth(n_ng, edges_ng)
            if girth_ng >= g:
                return n_ng, edges_ng
    
    # Fallback to LUW construction
    return _luw_construct(k, g)

def _construct_norm_graph(q):
    # Vertices are elements of F_{q^2}
    # q=4, F_16
    # We represent F_16 as pairs (a, b) in F_4 x F_4
    # F_4 = {0, 1, 2, 3} with x^2 + x + 1 = 0
    # Multiplication table for F_4
    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def add_f4(a, b):
        return a ^ b
        
    def mul_f4(a, b):
        return table[a][b]
        
    def add_f16(a, b):
        return (add_f4(a[0], b[0]), add_f4(a[1], b[1]))
        
    def mul_f16(a, b):
        # (a0 + a1*w) * (b0 + b1*w) = a0b0 + (a0b1 + a1b0)w + a1b1*w^2
        # w^2 = w + 1
        # = a0b0 + a1b1 + (a0b1 + a1b0 + a1b1)w
        real = add_f4(mul_f4(a[0], b[0]), mul_f4(a[1], b[1]))
        imag = add_f4(add_f4(mul_f4(a[0], b[1]), mul_f4(a[1], b[0])), mul_f4(a[1], b[1]))
        return (real, imag)
        
    def norm_f16(a):
        # N(a) = a * a^q = a * a^4
        # In F_16, x^4 is the Frobenius automorphism
        # (a0 + a1*w)^4 = a0^4 + a1^4 * w^4
        # In F_4, x^4 = x
        # w^4 = (w^2)^2 = (w+1)^2 = w^2 + 2w + 1 = (w+1) + 0 + 1 = w
        # So w^4 = w
        # Thus (a0 + a1*w)^4 = a0 + a1*w
        # Wait, in F_4, x^2 = x+1 for x!=0?
        # 1^2=1, 2^2=4=0? No.
        # F_4 elements: 0, 1, w, w+1.
        # 1^4=1.
        # w^2 = w+1. w^4 = (w+1)^2 = w^2+2w+1 = (w+1)+1 = w.
        # (w+1)^2 = w. (w+1)^4 = w^2 = w+1.
        # So x^4 = x for all x in F_4.
        # Thus a^4 = a.
        # N(a) = a * a = a^2.
        # So Norm is just the square in F_16?
        # N(a) = a^5.
        # a^5 = a^4 * a = a * a = a^2.
        # So N(a) = a^2.
        
        return mul_f16(a, a)
        
    # Vertices 0..255
    # Map index to F_16 element
    # a = (a0, a1)
    # index = a0 * 4 + a1
    
    def idx_to_el(i):
        return (i // 4, i % 4)
        
    def el_to_idx(e):
        return e[0] * 4 + e[1]
        
    # Precompute neighbors
    # x ~ y if N(x-y) = 1
    # 1 in F_16 is (1, 0)
    one = (1, 0)
    
    # Find all z such that N(z) = 1
    # z^2 = 1
    # (z0 + z1*w)^2 = 1
    # z0^2 + z1^2 w^2 + 2 z0 z1 w = 1
    # z0^2 + z1^2 (w+1) + 0 = 1
    # z0^2 + z1^2 + z1^2 w = 1
    # Real part: z0^2 + z1^2 = 1
    # Imag part: z1^2 = 0 => z1 = 0
    # So z0^2 = 1 => z0 = 1 (in F_4, 1^2=1, 3^2=1? 3=w+1, (w+1)^2=w. No.)
    # In F_4: 0^2=0, 1^2=1, 2^2=0? No.
    # 2 = w+1? No, 2 is usually w+1 in some bases.
    # Let's use the table.
    # table[1][1]=1, table[2][2]=3, table[3][3]=2.
    # So squares are 0, 1, 3, 2.
    # We need z1=0.
    # z0^2 = 1.
    # z0=1 works.
    # Are there other solutions?
    # N(z)=1 means z^5=1.
    # The multiplicative group of F_16 is cyclic of order 15.
    # Solutions to z^5=1 are the 5th roots of unity.
    # There are 5 solutions.
    
    roots = []
    for i in range(16):
        z = idx_to_el(i)
        if norm_f16(z) == one:
            roots.append(i)
            
    # Build edges
    edges = []
    for x_idx in range(16):
        x = idx_to_el(x_idx)
        for r_idx in roots:
            r = idx_to_el(r_idx)
            # y = x + r
            y = add_f16(x, r)
            y_idx = el_to_idx(y)
            if x_idx < y_idx:
                edges.append((x_idx, y_idx))
                
    return 16, edges

def _compute_girth(n, edges):
    # BFS from each vertex to find shortest cycle
    # This is O(n * (n+m)) which is fine for n=16
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    return min_girth if min_girth != float('inf') else 0

def _luw_construct(k, g):
    # Original LUW code
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
