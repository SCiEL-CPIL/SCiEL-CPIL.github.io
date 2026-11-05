import time
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = float('inf')
    best_edges = None
    
    # Candidate 1: LUW construction with variable d
    if k == 4:
        # Try d values. The girth of CD(d, q) is roughly 2d+4.
        # For g=10, we might need d=3 or d=4.
        # Current code used d=6. Let's try d from 3 to 6.
        for d in range(3, 7):
            if time.time() > deadline - 20:
                break
            try:
                n_luw, edges_luw = _construct_luw_d(k, g, d)
                if n_luw < best_n and _verify_girth_fast(n_luw, edges_luw, k, g):
                    best_n = n_luw
                    best_edges = edges_luw
            except Exception:
                pass
    
    # Candidate 2: SL(2,q) Cayley graph
    if k == 4:
        for q in [5, 7, 9, 11, 13]:
            if time.time() > deadline - 20:
                break
            try:
                n_cay, edges_cay = _construct_sl2_cayley(q)
                if n_cay < best_n and _verify_girth_fast(n_cay, edges_cay, k, g):
                    best_n = n_cay
                    best_edges = edges_cay
            except Exception:
                pass
                
    # Fallback
    if best_edges is None:
        try:
            best_n, best_edges = _construct_luw_d(k, g, 6)
        except Exception:
            best_n = k
            best_edges = []
            
    return best_n, best_edges

def _construct_luw_d(k, g, d):
    """Construct LUW graph with specific d."""
    q = k
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int32)
    else:
        table = None

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    index = {}
    order = []
    edges = []
    
    start = (0, tuple([0] * (d + 1)))
    index[start] = 0
    order.append(start)
    
    def get_neighbours(vertex):
        side, x = vertex
        x = np.array(x, dtype=np.int32)
        neighbours = []
        for first in range(q):
            y = np.zeros(d + 1, dtype=np.int32)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    if q == 4:
                        y[c] = (x[c] + table[y[a], x[b]]) % q
                    else:
                        y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    if q == 4:
                        y[c] = (x[c] - table[x[a], y[b]]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
            neighbours.append((1 - side, tuple(y)))
        return neighbours
    
    for vertex in order:
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _verify_girth_fast(n, edges, k, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
            
        while queue:
            u, p, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if v == start:
                    if depth + 1 < g:
                        return False
                    continue
                state = (v, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, u, depth + 1))
                    
    return True

def _construct_sl2_cayley(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        inv_table = {0:0, 1:1, 2:3, 3:2}
        def inv_func(x): return inv_table[x]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        def inv_func(x):
            if x == 0: return 0
            for i in range(1, q):
                if (x * i) % q == 1:
                    return i
            return 0

    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            add(mul(a1, a2), mul(b1, c2)),
            add(mul(a1, b2), mul(b1, d2)),
            add(mul(c1, a2), mul(d1, c2)),
            add(mul(c1, b2), mul(d1, d2))
        )

    def mat_inv(m):
        a, b, c, d = m
        det = sub(mul(a, d), mul(b, c))
        if det == 0:
            return None
        det_inv = inv_func(det)
        return (
            mul(det_inv, d),
            mul(det_inv, sub(0, b)),
            mul(det_inv, sub(0, c)),
            mul(det_inv, a)
        )

    A = (0, sub(0, 1), 1, 0)
    B = (1, 1, 0, 1)
    A_inv = mat_inv(A)
    B_inv = mat_inv(B)
    
    gens = [A, A_inv, B, B_inv]
    
    start = (1, 0, 0, 1)
    index = {start: 0}
    order = [start]
    edges = []
    
    for v in order:
        idx_v = index[v]
        for gen in gens:
            w = mat_mul(v, gen)
            if w not in index:
                index[w] = len(order)
                order.append(w)
            idx_w = index[w]
            if idx_v < idx_w:
                edges.append((idx_v, idx_w))
                
    return len(order), edges
