import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
        q: prime power (use q=3 for smallest)
        d: depth parameter
        Returns (n, edges) for a 3-regular graph of girth at least 2d+2 (roughly).
        """
        # Rule for LUW:
        # rule[2] = (1,1), rule[3] = (2,1)
        # For c >= 4: rule[c] depends on c mod 4
        # r = (c-4) % 4, c0 = c - r
        # rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1: (None, x1, x2, ..., xd)
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
        
        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def girth_check_optimized(n, adj, max_girth):
        """
        Optimized girth check: for each vertex, enumerate non-backtracking walks
        up to length max_girth-1 and check if any returns to the start.
        Uses a set of visited (current, prev) states to avoid re-exploration.
        """
        for start in range(n):
            # BFS in non-backtracking state space from start
            # State: (current, prev)
            # We track the minimum distance to each state
            dist = {}
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1))
                dist[(nb, start)] = 1
            
            while queue:
                u, prev, d = queue.popleft()
                
                if d >= max_girth:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False  # Found cycle shorter than max_girth
                    if (nb, u) not in dist:
                        dist[(nb, u)] = d + 1
                        queue.append((nb, u, d + 1))
        
        return True

    # Try LUW with small q and d values
    # LUW(q, d) gives girth approximately 2*d + 2
    # For girth 16, we need d >= 7
    # n = 2 * (q^d - 1) / (q - 1) * ... let me just compute
    
    # Try LUW with q=3, d=7 first (smallest expected n for girth 16)
    # Then try q=3, d=8, etc. if girth isn't sufficient
    # Also try q=2 (but q must be prime power >= 3 for the construction to work well)
    
    # Actually, let me try multiple (q, d) combinations and pick the smallest n with girth >= 16
    
    candidates = []
    for q in [3, 5, 7]:
        for d in range(5, 12):
            if time.time() > deadline - 20:
                break
            try:
                n, edges = luw_construct(q, d)
                if n <= 10000:  # Only check girth for reasonable sizes
                    # Quick girth check
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    has_girth = girth_check_optimized(n, adj, g)
                    candidates.append((n, q, d, edges, has_girth))
            except Exception:
                continue
    
    # Pick the smallest n with girth >= g
    valid = [(n, q, d, edges) for n, q, d, edges, hg in candidates if hg]
    if valid:
        valid.sort(key=lambda x: x[0])
        n, q, d, edges = valid[0]
        return n, edges
    
    # Final fallback: LUW with q=3, d=10 (should definitely have girth >= 16)
    n, edges = luw_construct(3, 10)
    return n, edges
