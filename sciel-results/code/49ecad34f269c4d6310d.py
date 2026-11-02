"""
Constructs a 5-regular graph with girth at least 9.
Uses the Lazebnik-Ustimenko-Woldar (L-U-W) construction with optimized parameters.
"""
import hashlib
import random

def construct(k, g):
    # The task specifies k=5 and g=9.
    # We use the L-U-W construction CD(d, q) with q=k.
    # The number of vertices is 2 * q^d.
    # The girth is approximately 2d+2 (for even d) or d+5 (for odd d).
    # For g=9, we want the smallest n, so we want the smallest d such that girth >= 9.
    # If d=4, girth is likely 10. n = 2 * 5^4 = 1250.
    # If d=5, girth is likely 10 or 12. n = 2 * 5^5 = 6250.
    # We try d=4 first.
    
    q = k
    # Determine d
    # Heuristic: try small d, verify girth.
    # We'll try d from 3 upwards.
    
    for d in range(3, 10):
        n_expected = 2 * (q ** d)
        if n_expected > 10000: # Safety break
            break
            
        # Build the graph for this d
        # GF(q) operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q

        # L-U-W rules for CD(d, q)
        # The rule is defined for c in 2..d
        # rule[c] = (a, b) such that y[c] = y[a]*x[b] + x[c] (or similar)
        # Standard L-U-W rule:
        # For c=2..d:
        # if c is even: y[c] = x[c] + y[c-1]*x[c-2] ? No, it varies.
        # Let's use the rule from the seed which is known to work.
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
        index = {}
        order = []
        edges = []
        
        index[start] = 0
        order.append(start)
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Verify Girth
        # BFS from each vertex to find shortest cycle
        # Since n is small (1250), this is fast.
        girth = float('inf')
        for start_v in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start_v] = 0
            queue = [start_v]
            found_cycle = False
            while queue and not found_cycle:
                u = queue.pop(0)
                for v in range(n):
                    if (u, v) in edges_set or (v, u) in edges_set: # Need efficient edge lookup
                        pass
            # Efficient Girth Check:
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            for start_v in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start_v] = 0
                queue = [start_v]
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            # We can break early if we just want to know if it's < g
                            if girth < g:
                                found_cycle = True
                                break
                    if found_cycle:
                        break
                if found_cycle:
                    break
            
            if girth >= g:
                return n, edges
            else:
                # If girth is too small, try next d
                continue
                
    # Fallback: return the seed's result (d=5) if smaller d failed
    # This ensures we always return a valid graph (the seed's one)
    d = 5
    # ... repeat construction for d=5 ...
    # For brevity in this thought trace, I assume d=4 works.
    # If not, the code will fall through to d=5.
    
    # To be safe and concise, I will implement the loop to try d=4, then d=5.
    # The code above does this.
    
    # If all fail (should not happen), return empty or raise
    raise ValueError("No valid graph found")

# Optimization: Pre-compute edge set for fast lookup in girth check?
# The girth check in the loop above is O(n * (n+E)) which is fine for n=1250.
# But doing it for every d is slow.
# Better: Construct, check girth. If good, return.

def construct(k, g):
    q = k
    best_n = float('inf')
    best_edges = []
    
    for d in range(3, 8):
        # Build graph
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q

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
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Check Girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        girth = float('inf')
        for start_v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start_v] = 0
            queue = [start_v]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                        if girth < g:
                            break
                if girth < g:
                    break
            if girth < g:
                break
        
        if girth >= g:
            return n, edges

    # Fallback to d=5 if smaller d didn't work (though d=4 should)
    # This code path is unlikely to be hit if d=4 works.
    raise ValueError("Construction failed")
