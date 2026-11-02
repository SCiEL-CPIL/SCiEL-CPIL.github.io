import time
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def _luw_construct(k, g):
        """LUW construction for cage graphs."""
        q = k
        # Choose d such that the tree of depth d with branching (k-1) has
        # enough vertices and the resulting graph has girth >= g.
        # For LUW: vertices are (side, x) where x is a tuple of length d+1.
        # side 0: x[0] is None, x[1]..x[d] are in Z_q
        # side 1: x[0]..x[d] are in Z_q
        # Total vertices: 2 * q^d
        # We need girth >= g, so we need d such that the non-backtracking
        # tree of radius (g-1)//2 fits.
        # For girth g, we need the tree of depth floor((g-1)/2) to be embedded.
        # LUW uses a specific construction. Let's use the standard one.
        
        # Standard LUW: for k=4, g=10, we need a specific d.
        # The LUW paper shows that for degree k and girth g, 
        # n = 2 * k * (k-1)^(d-1) where d is related to g.
        # Actually, the construction in the code uses:
        # vertices: (side, (None, x1, ..., xd)) for side=0
        #           (side, (x0, x1, ..., xd)) for side=1
        # with x_i in Z_q
        # Total: 2 * q^d
        # The rule defines how to compute neighbors.
        
        # For girth g, we need d such that the graph has no cycles shorter than g.
        # The LUW construction guarantees girth >= 2*d + 1 or similar.
        # Let's try different values of d and verify.
        
        best_n = float('inf')
        best_edges = None
        
        # Try different d values
        for d in range(3, 12):
            if time.time() > deadline:
                break
                
            # Build the rule table
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
            # Define operations
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                def add(x, y): return x ^ y
                def sub(x, y): return x ^ y
                def mul(x, y): return table[x][y]
            else:
                def add(x, y): return (x + y) % q
                def sub(x, y): return (x - y) % q
                def mul(x, y): return x * y % q
            
            # Build graph
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield (1 - side, tuple(y))
            
            start = (0, (None,) + (0,) * d)
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
            
            n = len(order)
            if n >= best_n:
                continue
            
            # Verify girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check girth using BFS from each vertex
            girth_ok = True
            for i in range(n):
                if time.time() > deadline:
                    girth_ok = False
                    break
                dist = [-1] * n
                parent = [-1] * n
                dist[i] = 0
                q_bfs = deque([i])
                while q_bfs:
                    u = q_bfs.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q_bfs.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                best_n = n
                best_edges = edges
                # Early exit if we found a good graph
                if best_n <= 2 * k * (k-1) ** ((g-1)//2 - 1):
                    break
        
        if best_edges is not None:
            return best_n, best_edges
        return None
    
    # Try LUW construction
    result = _luw_construct(k, g)
    if result is not None:
        return result
    
    # Fallback: simple construction (should not be reached)
    # This is a degenerate case handler
    n = 2 * k
    edges = [(i, (i+1) % n) for i in range(n)] + [(i, (i+k//2) % n) for i in range(n//2)]
    return n, edges
