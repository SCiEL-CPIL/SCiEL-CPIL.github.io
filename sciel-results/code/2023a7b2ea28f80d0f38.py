"""Construct a 5-regular graph with girth at least 9 (cage problem n(5,9)).
Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for prime q=k.
For k=5, g=9: d=4 (since g-5=4 is even, use g-4=5? No: g=9, g-5=4 even -> d=5).
Actually: d = g - 5 if odd, else g - 4. 9-5=4 even -> d=9-4=5.
The LUW graph CD(q, d) has girth >= d+5 = 10 >= 9. It is q-regular.
For q=5, d=5, the connected component of zero has n vertices and is valid.
"""

def construct(k, g):
    q = k
    
    def build_luw(d):
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
        
        rule = {}
        for c in range(2, d + 1):
            if c <= 4:
                rule[c] = {2: (1, 1), 3: (2, 1), 4: (1, 2)}[c]
            else:
                r = (c - 5) % 4
                c0 = c - r
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
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    def check_girth(n, edges, g):
        """Check if graph has no cycle shorter than g using non-backtracking walks."""
        if n < 3:
            return True
            
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # For each vertex, do BFS with non-backtracking constraint
        # Track (current, previous) pairs to avoid backtracking
        # If we return to start at depth < g, we found a short cycle
        
        import sys
        sys.setrecursionlimit(100000)
        
        for start in range(n):
            # visited[(curr, prev)] = True means we've been at curr coming from prev
            visited = set()
            # Queue of (current, previous, depth)
            from collections import deque
            queue = deque()
            queue.append((start, -1, 0))
            visited.add((start, -1))
            
            while queue:
                curr, prev, depth = queue.popleft()
                
                # If we're back at start and depth >= 3 (cycle length)
                if curr == start and depth > 0:
                    if depth < g:
                        return False
                    continue
                
                if depth >= g - 1:
                    continue
                    
                for next_v in adj[curr]:
                    if next_v == prev:
                        continue  # Non-backtracking
                    if (next_v, curr) in visited:
                        continue
                    visited.add((next_v, curr))
                    queue.append((next_v, curr, depth + 1))
                    
        return True

    # Moore bound for k=5, g=9 (odd girth 2r+1 where r=4):
    # n >= k * ((k-1)^r - 1) / (k-2) = 5 * (4^4 - 1) / 3 = 5 * 255 / 3 = 425
    
    moore_bound = k * ((k-1)**((g-1)//2) - 1) // (k-2)
    
    # Try d values. n = 2 * q^d for LUW.
    # We want smallest n >= moore_bound that is valid.
    # d=4: n=1250, d=5: n=6250
    # Check if d=4 is valid first (smaller)
    
    import time
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try d=4 first
    n4, edges4 = build_luw(4)
    if time.time() > deadline:
        # Time out, return d=5 as fallback (known valid)
        n5, edges5 = build_luw(5)
        return n5, edges5
        
    if check_girth(n4, edges4, g):
        return n4, edges4
    else:
        # d=4 not valid, use d=5
        n5, edges5 = build_luw(5)
        return n5, edges5
