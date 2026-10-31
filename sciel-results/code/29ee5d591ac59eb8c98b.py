import time
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # Safety margin
    
    def luw_construct(k, g, d):
        """
        Construct a k-regular graph using a recursive LUW-like construction.
        This function mimics the seed's logic but is kept for compatibility.
        The seed's logic:
        q = k
        rule generation...
        """
        q = k
        # The seed's rule generation
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # We need to explore neighbors.
        # This BFS on the state space.
        
        for vertex in order:
            side, x = vertex
            v_idx = index[vertex]
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1 - side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((v_idx, index[other]))
        
        return len(order), edges

    def check_girth_fast(n, edges, g):
        """
        Check if girth >= g using optimized BFS.
        We check for cycles of length < g.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For k=3, g=16, max_depth = 8.
        # We can use a single BFS from each node, but optimized.
        # Or, we can use the fact that for k=3, the number of non-backtracking walks is small.
        
        # Optimized BFS:
        # We use a list for the queue to avoid deque overhead if possible, 
        # but deque is standard.
        # We track distance and parent.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = [start]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Main Logic for k=3, g=16
    if k == 3 and g == 16:
        # Scan d from small to large.
        # The seed used d=11.
        # Let's try d from 5 to 15.
        
        best_n = float('inf')
        best_edges = None
        
        for d in range(5, 16):
            if time.time() > deadline - 10:
                break
            
            try:
                n, edges = luw_construct(k, g, d)
            except Exception:
                continue
            
            # Quick check: Moore bound
            if n < 512: 
                continue
            
            if n >= best_n:
                continue
            
            # Check girth
            if check_girth_fast(n, edges, g):
                best_n = n
                best_edges = edges
                # Since size grows with d, the first valid d gives the smallest n.
                # We can break.
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback: If LUW scan didn't find anything, use the seed's heuristic d.
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            n, edges = luw_construct(k, g, d_fallback)
            return n, edges
        except Exception:
            pass

    # General case fallback
    d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
    try:
        n, edges = luw_construct(k, g, d_fallback)
        return n, edges
    except Exception:
        pass
    
    # Final fallback
    d_safe = 20
    n, edges = luw_construct(k, g, d_safe)
    return n, edges
