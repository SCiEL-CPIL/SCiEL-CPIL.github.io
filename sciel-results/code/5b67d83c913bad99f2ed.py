import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s
    
    # Helper to check if a graph is valid (k-regular, girth >= g)
    # Returns (is_valid, girth_found)
    def check_graph(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False, 0
            
        # Check girth using BFS from each vertex (non-backtracking)
        # For k=3, we can use a faster method: BFS from each vertex, 
        # tracking the parent to avoid backtracking.
        # Girth is the shortest cycle.
        min_girth = float('inf')
        
        for s in range(n):
            # dist[v] = distance from s
            # parent[v] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle
                        # Cycle length is dist[u] + dist[v] + 1
                        # But this might be a "chord" not a simple cycle?
                        # In a tree BFS, if we see a visited node that is not parent,
                        # it forms a cycle. The length is dist[u] + dist[v] + 1.
                        # However, this is only the length of the cycle formed by the 
                        # two tree paths and the edge (u,v).
                        # This is a valid cycle.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            # Optimization: if min_girth is already < g, we can stop early?
            # No, we need the exact girth to certify it's >= g.
            # But if we find a cycle < g, we can return False immediately.
            if min_girth < g:
                return False, min_girth
        
        return True, min_girth

    # Strategy: Try specific small constructions
    # 1. Generalized Petersen Graphs GP(n, k)
    # 3-regular, 2n vertices.
    # Girth of GP(n, k) is known for many cases.
    # We want girth >= 16.
    # For GP(n, k), girth is usually small (3, 5, 6, 8, 10...).
    # GP(15, 2) has girth 10? GP(21, 4)?
    # Let's try a few candidates.
    
    candidates = []
    
    # Add some known or potential candidates
    # GP(n, k)
    for n in range(15, 100):
        for k in range(1, n//2):
            if gcd(n, k) != 1: continue
            # Simple check: no triangles
            # GP(n,k) has triangles if k=1 or k=n-1 (which is k=1 effectively)
            # or if 2k = n (even n).
            if k == 1 or k == n-1: continue
            if 2*k == n: continue
            # Check 5-cycles? Hard to do analytically quickly.
            # We will construct and check.
            candidates.append((n, k))
            
    # Shuffle to try diverse ones
    random.shuffle(candidates)
    
    best_n = float('inf')
    best_edges = None
    
    # Try LUW first as a fallback baseline if we don't find better quickly?
    # The current program returns LUW. Let's see if we can beat it.
    # LUW for k=3, g=16 likely has n > 512.
    # Moore bound for k=3, g=16 is 2 + 3 * (1 + 2 + ... + 2^7) = 2 + 3*(2^8 - 1) = 2 + 765 = 767?
    # Wait, Moore bound for odd girth g=2r+1: 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # g=16 is even. Moore bound for even girth g=2r: 2 * sum_{i=0}^{r-1} (k-1)^i
    # r=8. 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 2 * 255 = 510.
    # So n >= 510.
    # Current LUW likely produces something around 1000+.
    # If we find a graph with n < LUW_n, we improve.
    
    # Let's generate GP graphs and check
    checked = 0
    for n, k in candidates:
        if time.time() > deadline - 10:
            break
            
        # Construct GP(n, k)
        # Vertices: (0..n-1, 0) outer, (0..n-1, 1) inner
        # Edges:
        # Outer: (i, 0) - ((i+1)%n, 0)
        # Inner: (i, 1) - ((i+k)%n, 1)
        # Spokes: (i, 0) - (i, 1)
        
        edges = []
        for i in range(n):
            edges.append((i, (i+1)%n)) # Outer, using 0..n-1
            edges.append((n+i, n+(i+k)%n)) # Inner
            edges.append((i, n+i)) # Spoke
            
        # Check validity
        # We can do a quick check for small cycles first to reject fast
        # But check_graph does full BFS.
        # Let's do a quick triangle check?
        # GP(n,k) has no triangles if k not 1, n-1 and n not 2k.
        
        # Quick 5-cycle check?
        # Just run the full check but limit it?
        # The full check is O(n * E) which is fine for n~100.
        
        is_valid, girth = check_graph(2*n, edges)
        if is_valid and girth >= g:
            if 2*n < best_n:
                best_n = 2*n
                best_edges = edges
                # Update deadline if we find a good one? No, keep searching for smaller.
                # But if we find one, we have a valid result.
                
        checked += 1
        
    # If we found a GP graph, return it
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW Construction (from current program)
    # This ensures we return a valid result even if GP search fails
    def luw_construct(k, g):
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
                    
        return len(order), edges

    return luw_construct(k, g)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
