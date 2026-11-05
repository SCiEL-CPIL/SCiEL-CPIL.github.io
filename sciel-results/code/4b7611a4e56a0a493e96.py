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
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < g
        # Using non-backtracking walks
        # For each start vertex, BFS up to depth g-1
        # If we return to start at depth L < g, cycle found.
        
        import sys
        sys.setrecursionlimit(10000)
        
        for start in range(n):
            # visited[node] = min_depth_reached
            # We only care about cycles back to start
            # Non-backtracking: (prev, curr)
            # We can use a set of (curr, prev) visited?
            # Simpler: BFS from start, tracking (curr, prev)
            # If we hit (start, prev) where prev is a neighbor of start, and depth >= 3, we have a cycle.
            # Actually, if we hit start at depth L, it's a cycle of length L.
            
            visited = set()
            # (curr, prev, depth)
            stack = [(start, -1, 0)]
            while stack:
                curr, prev, depth = stack.pop()
                if depth > 0 and curr == start:
                    if depth < g:
                        return False
                    continue
                if (curr, prev) in visited:
                    continue
                visited.add((curr, prev))
                
                if depth >= g - 1:
                    continue
                    
                for next_v in adj[curr]:
                    if next_v != prev:
                        stack.append((next_v, curr, depth + 1))
                        
        return True

    # Try smaller d first to minimize n
    # d must be such that girth >= g
    # For odd d, girth >= d+5.
    # For even d, girth is not guaranteed, but we can check.
    # We want smallest n = 2 * q^d. So smallest d.
    
    # Candidate d values
    # g=9.
    # d=4: n=1250. Check if valid.
    # d=5: n=6250. Guaranteed valid (girth>=10).
    
    candidates = []
    # Try d=4, 3, 2... down to 1?
    # d=1: n=10. Girth?
    # d=2: n=50.
    # d=3: n=250.
    # d=4: n=1250.
    
    # We need to be careful with time.
    # Check d=4 first. If valid, return.
    # If not, try d=3, 2, 1?
    # If any are valid, they are smaller.
    
    # However, if d is too small, girth is likely < g.
    # Checking girth for n=1250 is fast.
    # Checking for n=250 is very fast.
    
    # Let's try d=4, 3, 2, 1 in that order?
    # Or just d=4 and d=5?
    # d=4 is the most likely candidate to be valid and small.
    
    # To be safe and fast, try d=4.
    # If valid, return.
    # Else return d=5.
    
    # What if d=3 is valid? n=250.
    # Let's try d=4, 3, 2, 1.
    # Time limit 120s.
    # Girth check for n=1250 is ~20M ops. ~1-2s.
    # Girth check for n=250 is ~0.4M ops. <0.1s.
    # So we can try all.
    
    import time
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_n = None
    best_edges = None
    
    for d in [4, 3, 2, 1]:
        if time.time() > deadline:
            break
        n, edges = build_luw(d)
        # Quick check: if n < Moore bound, it's definitely invalid?
        # Moore bound for k=5, g=9 is 428.
        # If n < 428, it cannot have girth 9.
        # So we can skip girth check for n < 428?
        # Yes, by definition, if n < Moore bound, no such graph exists.
        # So if d=1 (n=10), d=2 (n=50), d=3 (n=250), they are all < 428.
        # So they are INVALID.
        # We can skip them entirely!
        
        # Moore Bound calculation:
        # For odd girth g=2r+1:
        # n >= k + k(k-1) + ... + k(k-1)^r
        # r = (9-1)/2 = 4.
        # n >= 5 + 5*4 + 5*4^2 + 5*4^3 + 5*4^4
        # n >= 5 + 20 + 80 + 320 + 1280 = 1705?
        # Wait, the Moore bound formula for odd girth:
        # n >= k * ( (k-1)^{(g-1)/2} - 1 ) / (k-2)
        # k=5, g=9. (g-1)/2 = 4.
        # n >= 5 * (4^4 - 1) / 3 = 5 * 255 / 3 = 5 * 85 = 425.
        # So Moore bound is 425.
        
        if n < 425:
            continue
            
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            break # Found smallest valid d (since we iterate 4, 3, 2, 1 and n grows with d, wait. n=2*q^d. Smaller d means smaller n. So we should iterate 1, 2, 3, 4. But 1,2,3 are < 425. So only d=4 is candidate before d=5.)
            
    if best_n is not None:
        return best_n, best_edges
    else:
        # Fallback to d=5
        n, edges = build_luw(5)
        return n, edges
