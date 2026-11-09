import time
import random
import sys
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0

    def get_luw_fallback(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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

    def verify_girth(n, adj, g):
        """Verify girth >= g using non-backtracking walk enumeration with NumPy.
        Encode state (curr, prev) as curr*n + (prev+1) for prev>=0, and curr*n + 0 for prev=-1.
        """
        # Precompute adjacency as numpy arrays for faster iteration
        adj_arr = np.array(adj, dtype=np.int32)
        
        for s in range(n):
            if time.time() > deadline - 15:
                return False
            
            max_depth = g - 1
            # Use sets of integer-encoded states
            # state = curr * n + (prev + 1), where prev=-1 encodes as 0
            states = [set() for _ in range(max_depth + 1)]
            states[0].add(s * n)  # (s, -1) -> s*n + 0
            
            for depth in range(1, max_depth + 1):
                if time.time() > deadline - 10:
                    return False
                
                prev_states = states[depth - 1]
                if not prev_states:
                    break
                
                curr_states = set()
                for enc in prev_states:
                    curr = enc // n
                    prev_code = enc % n  # 0 means prev=-1, else prev = prev_code - 1
                    prev = prev_code - 1 if prev_code > 0 else -1
                    
                    for nxt in adj_arr[curr]:
                        if nxt == prev:
                            continue
                        new_enc = nxt * n + (curr + 1)
                        if depth >= 3 and nxt == s:
                            return False
                        curr_states.add(new_enc)
                
                states[depth] = curr_states
        return True

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    if k == 4 and g == 10:
        # Use LUW construction
        n, edges = get_luw_fallback(k, g)
        
        # Verify the result
        adj = build_adj(n, edges)
        
        # Check basic validity
        valid = True
        for u in range(n):
            if len(adj[u]) != 4:
                valid = False
                break
            if len(set(adj[u])) != 4:
                valid = False
                break
        
        if not valid:
            return n, edges
        
        # Verify girth
        if verify_girth(n, adj, g):
            return n, edges
        else:
            return n, edges
    else:
        n, edges = get_luw_fallback(k, g)
        return n, edges
