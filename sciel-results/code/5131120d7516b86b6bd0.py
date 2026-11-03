"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; evaluator kills at 150s, keep margin

    q = k
    d = max(3, (g + 1) // 2 - 1)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    def build_rule(d_val):
        rule = [None, None, (1,1), (2,1), (1,2)]
        for c in range(5, d_val+1):
            r = (c-5) % 4
            c0 = c - r
            rule.append([(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r])
        return rule
    
    def build_graph(d_val, rule):
        """Build the LUW graph using integer-encoded vertices for speed."""
        # Encode (side, x1,...,xd) as integer: side * q^d + x1*q^(d-1) + ... + xd
        qpow = [1] * (d_val + 1)
        for i in range(1, d_val + 1):
            qpow[i] = qpow[i-1] * q
        
        # Precompute rule operations for speed
        # For each c, we have (a, b) and the operation depends on side
        # side 0: y[c] = add(x[c], mul(y[a], x[b]))
        # side 1: y[c] = sub(x[c], mul(x[a], y[b]))
        
        index = {}
        order = []
        edges = []
        
        # Start vertex: side=0, x = (None, 0, 0, ..., 0) -> encode as 0 * q^d + 0 = 0
        # But we need to handle the None at position 1. Let's use a different encoding.
        # Actually, let's just use tuples but optimize the inner loop.
        
        start = (0, (None,) + (0,)*d_val)
        index[start] = 0
        order = [start]
        
        for vi, vertex in enumerate(order):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1-side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((vi, index[other]))
        
        return len(order), edges
    
    def verify_girth(n, edges, min_girth):
        """Verify girth >= min_girth using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([(start, 0, -1)])
            while queue:
                u, du, pu = queue.popleft()
                if du >= max_dist:
                    continue
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append((v, du + 1, u))
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    # Try initial depth
    rule = build_rule(d)
    n, edges = build_graph(d, rule)
    
    if verify_girth(n, edges, g):
        return n, edges
    
    # If girth check fails, try larger depths
    for d_try in range(d+1, d+6):
        if time.time() - t0 > deadline - 5:
            break
        rule_try = build_rule(d_try)
        n_try, edges_try = build_graph(d_try, rule_try)
        if verify_girth(n_try, edges_try, g):
            return n_try, edges_try
    
    return n, edges
# EVOLVE-BLOCK-END
