"""Improved program using L-U-W construction with optimized girth verification."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    
    # Precompute field operations for speed
    if q == 4:
        # GF(4) represented as 0,1,2,3 with XOR addition and table multiplication
        add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return add_tbl[x][y]
        def sub(x, y): return add_tbl[x][y] # In char 2, sub = add
        def mul(x, y): return mul_tbl[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    def build_graph(d):
        """Build L-U-W graph with given dimension d and return (n, edges, girth_ok)"""
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        
        # BFS to build graph
        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None, None, False
            vertex = order[i]
            i += 1
            v_side, v_x = vertex
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if v_side == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree regularity
        for v in range(n):
            if len(adj[v]) != k:
                return n, edges, False
        
        # Optimized girth verification using non-backtracking BFS
        # We only need to check up to depth g-2 (looking for cycles of length < g)
        # State: (current, prev), we track distance from start vertex
        # For each start vertex s, BFS from all neighbors of s
        
        max_depth = g - 2  # We're looking for cycles of length < g, so depth < g-1
        
        for s in range(n):
            if time.time() > deadline:
                return n, edges, False
            
            # visited[state] = distance from s
            visited = {}
            queue = deque()
            for nb in adj[s]:
                state = (nb, s)
                visited[state] = 1
                queue.append((state, 1))
            
            while queue:
                if time.time() > deadline:
                    return n, edges, False
                (cur, prev), dist = queue.popleft()
                
                # If we're back at s and total cycle length < g
                if cur == s:
                    if dist + 1 < g:
                        return n, edges, False
                    continue
                
                # Don't explore beyond needed depth
                if dist >= max_depth:
                    continue
                
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue  # non-backtracking
                    new_state = (nxt, cur)
                    if new_state not in visited:
                        visited[new_state] = dist + 1
                        queue.append((new_state, dist + 1))
        
        return n, edges, True

    # Try different values of d to find smallest valid graph
    best_n = None
    best_edges = None
    
    # The girth of L-U-W CD(d,q) is related to d
    # For girth g, we need d large enough
    # Try a range of d values
    d_candidates = []
    base_d = g - 5
    if base_d % 2 == 0:
        base_d = g - 4
    base_d = max(3, base_d)
    
    # Try several candidates around the expected value
    for d in range(max(3, base_d - 2), base_d + 4):
        d_candidates.append(d)
    
    # Sort by expected size (smaller d = smaller graph)
    d_candidates.sort()
    
    for d in d_candidates:
        if time.time() > deadline:
            break
        n, edges, ok = build_graph(d)
        if n is None:
            continue
        if ok and (best_n is None or n < best_n):
            best_n = n
            best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use original construction
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges, _ = build_graph(d)
    if n is not None:
        return n, edges
    
    # Last resort: return empty (shouldn't happen)
    return 0, []
# EVOLVE-BLOCK-END
