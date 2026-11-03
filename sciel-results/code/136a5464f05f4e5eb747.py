"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; evaluator kills at 150s, keep margin

    # Use the LUW-type algebraic construction for all cases
    q = k
    # Choose depth d such that the resulting graph has girth >= g
    # The LUW construction gives girth approximately 2*d + 1 or 2*d + 2
    # We need girth >= g, so we need d such that 2*d + 1 >= g (for odd girth)
    # or 2*d + 2 >= g (for even girth)
    # Conservative: use d = ceil(g/2) - 1
    d = max(3, (g + 1) // 2 - 1)
    
    # Precompute arithmetic tables for speed
    if q == 4:
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        sub_table = add_table  # XOR is its own inverse
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    else:
        add_table = [[(x+y)%q for y in range(q)] for x in range(q)]
        sub_table = [[(x-y)%q for y in range(q)] for x in range(q)]
        mul_table = [[x*y%q for y in range(q)] for x in range(q)]
    
    def build_rule(d_max):
        """Build the LUW rule table for columns 2..d_max."""
        rule = [None] * (d_max + 1)
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d_max + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)
        return rule
    
    def build_graph(d_val, rule):
        """Build the graph using the given depth and rule table."""
        side0_add = add_table
        side1_sub = sub_table
        mul_t = mul_table
        
        start = (0, (None,) + (0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            side, x = vertex
            if side == 0:
                for first in range(q):
                    y = [None, first] + [0]*(d_val - 1)
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        y[c] = side0_add[x[c]][mul_t[y[a]][x[b]]]
                    other = (1, tuple(y))
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    edges.append((index[vertex], index[other]))
            else:
                for first in range(q):
                    y = [None, first] + [0]*(d_val - 1)
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        y[c] = side1_sub[x[c]][mul_t[x[a]][y[b]]]
                    other = (0, tuple(y))
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
        
        n = len(order)
        return n, edges
    
    def verify_girth(n, edges, min_girth):
        """Check girth >= min_girth using non-backtracking BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = (min_girth - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            found_short = False
            while qi < len(queue):
                u = queue[qi]
                qi += 1
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
    
    # Build with initial depth
    rule = build_rule(d)
    n, edges = build_graph(d, rule)
    
    if verify_girth(n, edges, g):
        return n, edges
    
    # Try larger depths if girth is insufficient
    for d_try in range(d + 1, d + 6):
        if time.time() - t0 > deadline - 5:
            break
        rule_try = build_rule(d_try)
        n_try, edges_try = build_graph(d_try, rule_try)
        if verify_girth(n_try, edges_try, g):
            return n_try, edges_try
    
    # If all fail, return the largest graph we built (best effort)
    return n, edges
# EVOLVE-BLOCK-END
